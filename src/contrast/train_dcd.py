"""Feature-cache implementation of ICCV 2025 DCD.

This is a strict cache-level run: the frozen CLIP image embeddings are the
only input.  Dynamic class centers are updated in that feature space and the
loss mines a sparse hard subset using classification loss plus center
distance, while corrected trusted labels remain fully supervised.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from contrast.cache_experiment import (evaluate, load_cache_and_observed,
                                       make_run_dir, save_config, save_history,
                                       save_result)
from paths import clip_cache, observed_labels_cache


class LinearHead(nn.Module):
    def __init__(self, dim: int, classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initial_centers(features: torch.Tensor, observed: torch.Tensor,
                    true_labels: torch.Tensor, trusted: torch.Tensor,
                    classes: int) -> torch.Tensor:
    assignments = torch.where(trusted, true_labels, observed)
    centers = torch.zeros(classes, features.shape[1], dtype=features.dtype)
    fallback = features.mean(dim=0)
    for cls in range(classes):
        selected = features[assignments == cls]
        center = selected.mean(dim=0) if selected.shape[0] else fallback
        centers[cls] = F.normalize(center, dim=0)
    return centers


def update_centers(centers: torch.Tensor, features: torch.Tensor,
                   assignments: torch.Tensor, trusted: torch.Tensor,
                   confidence: torch.Tensor, momentum: float) -> None:
    with torch.no_grad():
        selected = trusted | ((~trusted) & (confidence >= 0.6))
        if not selected.any():
            return
        labels = assignments[selected].long()
        values = features[selected].float()
        sums = torch.zeros_like(centers).index_add_(0, labels, values)
        counts = torch.zeros(centers.shape[0], dtype=values.dtype)
        counts.index_add_(0, labels, torch.ones(labels.shape[0], dtype=values.dtype))
        valid = counts > 0
        means = sums[valid] / counts[valid].unsqueeze(1)
        means = F.normalize(means, dim=1)
        centers[valid].mul_(momentum).add_((1.0 - momentum) * means)
        centers[valid].copy_(F.normalize(centers[valid], dim=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--obs-labels", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hard-fraction", type=float, default=0.35)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache_path = args.cache or str(clip_cache(args.dataset))
    obs_path = args.obs_labels or str(observed_labels_cache(args.dataset))
    run_dir = make_run_dir(args.dataset, "dcd", args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_cache_and_observed(cache_path, obs_path)
    train_x, val_x, test_x = bundle["train_feats"], bundle["val_feats"], bundle["test_feats"]
    train_y, val_y, test_y = bundle["train_labels"], bundle["val_labels"], bundle["test_labels"]
    y_obs, trusted = bundle["y_obs"], bundle["s"].eq(0)
    classes = int(bundle["num_classes"])
    model = LinearHead(train_x.shape[1], classes).to(device)
    centers = initial_centers(train_x, y_obs, train_y, trusted, classes)
    loader = DataLoader(TensorDataset(train_x, train_y, y_obs, trusted),
                        batch_size=args.batch_size, shuffle=True, num_workers=2,
                        pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_acc, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, used = 0.0, 0
        for features, true_y, observed_y, is_trusted in loader:
            features = features.to(device, non_blocking=True)
            true_y = true_y.to(device)
            observed_y = observed_y.to(device)
            is_trusted = is_trusted.to(device)
            target = torch.where(is_trusted, true_y, observed_y)
            logits = model(features)
            per_sample_loss = F.cross_entropy(logits, target, reduction="none")
            confidence, predicted = logits.softmax(dim=-1).max(dim=-1)
            assignments = torch.where(is_trusted, true_y, predicted)
            with torch.no_grad():
                center = centers.to(device)[assignments]
                distance = 1.0 - F.cosine_similarity(features, center, dim=-1)
                score = per_sample_loss.detach() / (per_sample_loss.detach().mean() + 1e-6)
                score = 0.5 * score + 0.5 * distance / (distance.mean() + 1e-6)
                noisy = ~is_trusted
                if noisy.any():
                    noisy_scores = score[noisy]
                    threshold = torch.quantile(noisy_scores, 1.0 - args.hard_fraction)
                    hard = noisy & (score >= threshold)
                else:
                    hard = noisy
                selected = is_trusted | hard
                weights = (1.0 + score.detach()).clamp(max=4.0)
            if epoch <= args.warmup_epochs:
                # Preliminary semi-supervised stage before DCD hard mining.
                trusted_loss = per_sample_loss[is_trusted].mean() if is_trusted.any() else per_sample_loss.mean() * 0
                noisy_loss = per_sample_loss[noisy].mean() if noisy.any() else per_sample_loss.mean() * 0
                loss = trusted_loss + 0.25 * noisy_loss
            else:
                loss = (per_sample_loss[selected] * weights[selected]).sum() / weights[selected].sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            update_centers(centers, features.detach().cpu(), assignments.detach().cpu(),
                           is_trusted.detach().cpu(), confidence.detach().cpu(), args.center_momentum)
            running += float(loss.item())
            used += 1
        scheduler.step()
        val_metrics = evaluate(model, val_x, val_y, device)
        row = {"epoch": epoch, "train_loss": running / max(1, used), **val_metrics}
        history.append(row)
        print(f"[DCD][{args.dataset}] epoch={epoch:03d} val_acc={val_metrics['accuracy']*100:.2f}%")
        if val_metrics["accuracy"] > best_acc:
            best_acc, best_epoch, bad_epochs = val_metrics["accuracy"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "centers": centers, "epoch": epoch}, run_dir / "best.ckpt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    torch.save({"state_dict": model.state_dict(), "centers": centers, "epoch": epoch}, run_dir / "last.ckpt")
    save_history(run_dir, history)
    test_metrics = evaluate(model, test_x, test_y, device)
    save_config(run_dir, {"method": "DCD-cache-level", "dataset": args.dataset,
                          "cache": cache_path, "observed_labels": obs_path,
                          "args": vars(args), "trusted_ratio": float(trusted.float().mean())})
    save_result(run_dir, {"method": "DCD-cache-level", "dataset": args.dataset,
                          "best_val": history[max(0, best_epoch - 1)] if history else {},
                          "best_val_accuracy": best_acc, "best_epoch": best_epoch,
                          "test": test_metrics, "trusted_ratio": float(trusted.float().mean()),
                          "elapsed_seconds": time.time() - start_time})
    print(f"[DCD][DONE] {run_dir}")


if __name__ == "__main__":
    main()
