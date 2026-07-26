"""Cache-level adaptation of DLD (CVPR 2025).

Paper: ``Directional Label Diffusion Model for Learning from Noisy Labels``.
The original method diffuses labels while training an image classifier.  This
adaptation keeps the cached CLIP image features fixed and runs the directional
and random label-diffusion modules in label space, with a linear prediction
head used for validation and testing.  It is therefore directly comparable to
the other cache-only baselines in this project, but is not claimed to be a
pixel-level reproduction of the original training pipeline.
"""
from __future__ import annotations

import argparse
import math
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

from contrast.cache_experiment import (
    evaluate,
    load_cache_and_observed,
    make_run_dir,
    save_config,
    save_history,
    save_result,
)
from paths import clip_cache, observed_labels_cache


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prior_probabilities(bundle: dict, split: str, labels: torch.Tensor,
                        classes: int) -> torch.Tensor:
    prior = bundle.get(f"clip_probs_{split}")
    if isinstance(prior, torch.Tensor) and prior.ndim == 2 and prior.shape[0] == labels.shape[0]:
        if prior.shape[1] == classes:
            return prior.float().clamp_min(1e-6).div(prior.float().sum(1, keepdim=True).clamp_min(1e-6))
    embeds = bundle.get("clip_label_embeds")
    features = bundle[f"{split}_feats"]
    if isinstance(embeds, torch.Tensor) and embeds.ndim == 2 and embeds.shape[0] == classes:
        logits = F.normalize(features.float(), dim=-1) @ F.normalize(embeds.float(), dim=-1).t()
        return logits.mul(20.0).softmax(dim=-1).cpu()
    return F.one_hot(labels.clamp(0, classes - 1), classes).float()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time_value: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(torch.linspace(
            math.log(1.0), math.log(1000.0), half, device=time_value.device
        ))
        angles = time_value[:, None] * frequencies[None, :] * 2.0 * math.pi
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return F.pad(embedding, (0, 1)) if self.dim % 2 else embedding


class DLDCacheModel(nn.Module):
    def __init__(self, feature_dim: int, classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.feature_proj = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.label_proj = nn.Sequential(nn.Linear(2 * classes, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.time_embedding = SinusoidalTimeEmbedding(32)
        self.time_proj = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.direction = nn.Linear(hidden, classes)
        self.random = nn.Linear(hidden, classes)
        self.head = nn.Linear(feature_dim, classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor, noisy_label: torch.Tensor,
                prior: torch.Tensor, time_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.feature_proj(features)
        hidden = hidden + self.label_proj(torch.cat((noisy_label, prior), dim=-1))
        hidden = hidden + self.time_proj(self.time_embedding(time_value))
        hidden = self.trunk(hidden)
        return self.direction(hidden), self.random(hidden), self.head(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cache-level DLD adaptation.")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--obs-labels", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-weight", type=float, default=0.25)
    parser.add_argument("--align-weight", type=float, default=0.125)
    parser.add_argument("--align-temperature", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache_path = args.cache or str(clip_cache(args.dataset))
    obs_path = args.obs_labels or str(observed_labels_cache(args.dataset))
    run_dir = make_run_dir(args.dataset, "dld", args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_cache_and_observed(cache_path, obs_path)
    train_x = bundle["train_feats"]
    val_x, test_x = bundle["val_feats"], bundle["test_feats"]
    val_y, test_y = bundle["val_labels"], bundle["test_labels"]
    y_obs = bundle["y_obs"]
    trusted = bundle["s"].eq(0)
    classes = int(bundle["num_classes"])
    prior = prior_probabilities(bundle, "train", y_obs, classes)
    model = DLDCacheModel(train_x.shape[1], classes, args.hidden).to(device)
    loader = DataLoader(TensorDataset(train_x, y_obs, trusted, prior),
                        batch_size=args.batch_size, shuffle=True, num_workers=2,
                        pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_acc, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for features, observed, is_trusted, prior_batch in loader:
            features = features.to(device, non_blocking=True).float()
            observed = observed.to(device, non_blocking=True).long()
            is_trusted = is_trusted.to(device, non_blocking=True)
            prior_batch = prior_batch.to(device, non_blocking=True).float()
            y0 = F.one_hot(observed, classes).float()
            time_value = torch.rand(features.shape[0], device=device)
            direction_noise = prior_batch - y0
            random_noise = torch.randn_like(y0)
            y_t = y0 + time_value[:, None] * direction_noise + time_value[:, None] * random_noise
            y_direction, y_random, logits = model(features, y_t, prior_batch, time_value)
            clean_direction = F.mse_loss(y_direction, direction_noise)
            clean_random = F.mse_loss(y_random, random_noise)
            per_sample_ce = F.cross_entropy(logits, observed, reduction="none")
            # Trusted corrected labels are fully supervised; noisy labels retain
            # a small head loss while the diffusion target carries the signal.
            head_loss = torch.where(is_trusted, per_sample_ce, 0.25 * per_sample_ce).mean()
            with torch.no_grad():
                y0_hat = y_t - time_value[:, None] * y_direction - time_value[:, None] * y_random
                target = F.softmax(y0_hat / args.align_temperature, dim=-1)
            alignment = F.kl_div(
                F.log_softmax(logits / args.align_temperature, dim=-1), target,
                reduction="batchmean"
            ) * args.align_temperature ** 2
            loss = clean_direction + clean_random + args.head_weight * head_loss + args.align_weight * alignment
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())
        scheduler.step()
        model.eval()
        val_metrics = evaluate(model.head, val_x, val_y, device)
        row = {"epoch": epoch, "train_loss": running / max(1, len(loader)), **val_metrics}
        history.append(row)
        print(f"[DLD][{args.dataset}] epoch={epoch:03d} val_acc={val_metrics['accuracy'] * 100:.2f}%", flush=True)
        if val_metrics["accuracy"] > best_acc:
            best_acc, best_epoch, bad_epochs = val_metrics["accuracy"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, run_dir / "best.ckpt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    completed = history[-1]["epoch"] if history else 0
    torch.save({"state_dict": model.state_dict(), "epoch": completed}, run_dir / "last.ckpt")
    save_history(run_dir, history)
    test_metrics = evaluate(model.head, test_x, test_y, device)
    save_config(run_dir, {"method": "DLD-CVPR2025-cache-adapted", "dataset": args.dataset,
                          "cache": cache_path, "observed_labels": obs_path,
                          "args": vars(args), "trusted_ratio": float(trusted.float().mean())})
    save_result(run_dir, {
        "method": "DLD-CVPR2025-cache-adapted", "dataset": args.dataset,
        "paper": "Directional Label Diffusion Model for Learning from Noisy Labels",
        "best_val_accuracy": best_acc, "best_epoch": best_epoch,
        "test": test_metrics, "trusted_ratio": float(trusted.float().mean()),
        "epochs_completed": completed, "elapsed_seconds": time.time() - start_time,
    })
    print(f"[DLD][DONE] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
