"""Cache-level adaptation of IDO (NeurIPS 2025).

Paper: ``Handling Label Noise via Instance-Level Difficulty Modeling and
Dynamic Optimization``.  The implementation follows the paper's two-stage
structure: collect per-sample ``wrong event`` statistics with a base linear
classifier, fit a two-component probabilistic model to those statistics, then
train with instance-dependent dynamic weights and soft corrected targets.

Only frozen CLIP embeddings and observed training labels are used for model
updates.  This is a cache-level adaptation; it does not claim to reproduce the
paper's original image-backbone training pipeline.
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

from contrast.cache_experiment import (
    evaluate,
    load_cache_and_observed,
    make_run_dir,
    save_config,
    save_history,
    save_result,
)
from paths import clip_cache, observed_labels_cache


class LinearHead(nn.Module):
    def __init__(self, feature_dim: int, classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_labels(model: nn.Module, features: torch.Tensor,
                   device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    predictions = []
    for start in range(0, features.shape[0], batch_size):
        logits = model(features[start:start + batch_size].to(device, non_blocking=True))
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions)


def fit_wrong_event_mixture(rates: torch.Tensor, iterations: int = 50) -> tuple[torch.Tensor, dict]:
    """Fit a stable two-component Gaussian mixture to wrong-event rates.

    IDO models the wrong-event statistic probabilistically.  A two-component
    mixture is used here because it is dependency-free and remains stable for
    small datasets; the lower-mean component is interpreted as the cleaner
    component.
    """
    x = rates.float().clamp(1e-4, 1.0)
    q25, q75 = torch.quantile(x, torch.tensor([0.25, 0.75]))
    means = torch.stack((q25, q75)).clamp(1e-4, 1.0)
    variance = torch.full((2,), max(float(x.var().item()), 1e-3))
    mixture = torch.full((2,), 0.5)
    for _ in range(iterations):
        log_prob = torch.log(mixture.clamp_min(1e-6))[None, :] - 0.5 * (
            (x[:, None] - means[None, :]) ** 2 / variance[None, :]
            + torch.log(2.0 * torch.pi * variance)[None, :]
        )
        responsibility = log_prob.softmax(dim=1)
        counts = responsibility.sum(dim=0).clamp_min(1e-6)
        mixture = counts / counts.sum()
        means = (responsibility * x[:, None]).sum(dim=0) / counts
        variance = (responsibility * (x[:, None] - means[None, :]) ** 2).sum(dim=0) / counts
        variance = variance.clamp_min(1e-4)
    clean_component = int(means.argmin().item())
    clean_probability = responsibility[:, clean_component]
    return clean_probability, {
        "means": means.tolist(), "variance": variance.tolist(),
        "mixture": mixture.tolist(), "clean_component": clean_component,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cache-level IDO adaptation.")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--obs-labels", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache_path = args.cache or str(clip_cache(args.dataset))
    obs_path = args.obs_labels or str(observed_labels_cache(args.dataset))
    run_dir = make_run_dir(args.dataset, "ido", args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_cache_and_observed(cache_path, obs_path)
    train_x, val_x, test_x = bundle["train_feats"], bundle["val_feats"], bundle["test_feats"]
    val_y, test_y = bundle["val_labels"], bundle["test_labels"]
    y_obs = bundle["y_obs"]
    trusted = bundle["s"].eq(0)
    classes = int(bundle["num_classes"])
    loader = DataLoader(TensorDataset(train_x, y_obs), batch_size=args.batch_size,
                        shuffle=True, num_workers=2, pin_memory=True)
    start_time = time.time()

    # Stage 1: train a base classifier and collect wrong events over time.
    base = LinearHead(train_x.shape[1], classes).to(device)
    base_optimizer = torch.optim.AdamW(base.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
    wrong_events = torch.zeros(train_x.shape[0], dtype=torch.float32)
    for epoch in range(1, args.stage1_epochs + 1):
        base.train()
        for features, labels in loader:
            features = features.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            loss = F.cross_entropy(base(features), labels)
            base_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            base_optimizer.step()
        predictions = predict_labels(base, train_x, device, args.batch_size)
        wrong_events += predictions.ne(y_obs).float()
        print(f"[IDO][{args.dataset}] stage1={epoch:03d}/{args.stage1_epochs} "
              f"wrong_event_rate={wrong_events.mean().item() / epoch:.4f}", flush=True)

    rates = wrong_events / max(1, args.stage1_epochs)
    clean_probability, mixture_info = fit_wrong_event_mixture(rates)
    clean_probability[trusted] = 1.0
    torch.save({"wrong_events": wrong_events, "wrong_event_rates": rates,
                "clean_probability": clean_probability, "mixture": mixture_info},
               run_dir / "wrong_events.pt")

    # Stage 2: dynamic optimization using instance-level clean probabilities.
    model = LinearHead(train_x.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.stage2_epochs)
    stage2_dataset = TensorDataset(train_x, y_obs, clean_probability)
    stage2_loader = DataLoader(stage2_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    best_acc, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    for epoch in range(1, args.stage2_epochs + 1):
        model.train()
        running = 0.0
        for features, labels, clean_prob in stage2_loader:
            features = features.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            clean_prob = clean_prob.to(device, non_blocking=True).float()
            logits = model(features)
            probabilities = logits.softmax(dim=-1)
            observed_target = F.one_hot(labels, classes).float()
            # Clean samples keep their observed label; difficult/noisy samples
            # receive a detached model target and a lower dynamic weight.
            soft_target = clean_prob[:, None] * observed_target + (1.0 - clean_prob[:, None]) * probabilities.detach()
            dynamic_weight = (0.25 + 0.75 * clean_prob).clamp_min(0.05)
            loss_per_sample = -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            loss = (loss_per_sample * dynamic_weight).sum() / dynamic_weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())
        scheduler.step()
        val_metrics = evaluate(model, val_x, val_y, device)
        row = {"epoch": epoch, "train_loss": running / max(1, len(stage2_loader)), **val_metrics}
        history.append(row)
        print(f"[IDO][{args.dataset}] stage2={epoch:03d}/{args.stage2_epochs} "
              f"val_acc={val_metrics['accuracy'] * 100:.2f}%", flush=True)
        if val_metrics["accuracy"] > best_acc:
            best_acc, best_epoch, bad_epochs = val_metrics["accuracy"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "clean_probability": clean_probability}, run_dir / "best.ckpt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    completed = history[-1]["epoch"] if history else 0
    torch.save({"state_dict": model.state_dict(), "epoch": completed,
                "clean_probability": clean_probability}, run_dir / "last.ckpt")
    save_history(run_dir, history)
    test_metrics = evaluate(model, test_x, test_y, device)
    save_config(run_dir, {"method": "IDO-NeurIPS2025-cache-adapted", "dataset": args.dataset,
                          "cache": cache_path, "observed_labels": obs_path,
                          "args": vars(args), "trusted_ratio": float(trusted.float().mean()),
                          "mixture": mixture_info})
    save_result(run_dir, {
        "method": "IDO-NeurIPS2025-cache-adapted", "dataset": args.dataset,
        "paper": "Handling Label Noise via Instance-Level Difficulty Modeling and Dynamic Optimization",
        "best_val_accuracy": best_acc, "best_epoch": best_epoch,
        "test": test_metrics, "trusted_ratio": float(trusted.float().mean()),
        "wrong_event_mean": float(rates.mean()), "mixture": mixture_info,
        "epochs_completed": completed, "elapsed_seconds": time.time() - start_time,
    })
    print(f"[IDO][DONE] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
