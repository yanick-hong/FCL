"""Cache-level baselines required by the FCL revision.

The methods in this file use the same frozen CLIP cache, train/validation/test
splits, and observed-label cache as FCL.  Methods whose original papers rely
on image augmentations or end-to-end VLM tuning are explicitly marked as
cache-level adaptations in ``metrics.json``.

Available methods:
  - ``clip_conf_ce``: CLIP-confidence selection with standard CE.
  - ``random_filter_ce``: random selection at the same verification budget.
  - ``clip_zero_ce``: no human verification; CE on all CLIP pseudo-labels.
  - ``fcl_filter_ce``: FCL's exact filter with standard CE.
  - ``xie_trim_auc``: fixed-posterior trimmed pairwise AUC baseline inspired
    by Xie et al., TPAMI 2024, without FCL's model-posterior feedback.
  - ``fixmatch_cache`` and ``softmatch_cache``: cache-level SSL adaptations.
  - ``dividemix_cache``: loss-mixture cache-level DivideMix adaptation.
  - ``deft_cache``: CLIP dual-score/cache-level DeFT adaptation.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

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
    save_run_metadata,
)
from paths import clip_cache, observed_labels_cache


PAPER_INFO = {
    "clip_conf_ce": "CLIP confidence threshold + CE control",
    "random_filter_ce": "Random verification at the same budget + CE control",
    "clip_zero_ce": "Zero-human-verification CLIP pseudo-label CE lower bound",
    "fcl_filter_ce": "FCL filter + standard CE control",
    "xie_trim_auc": "Xie et al., Weakly Supervised AUC Optimization, TPAMI 2024 (trim adaptation)",
    "fixmatch_cache": "FixMatch cache-level adaptation",
    "softmatch_cache": "SoftMatch cache-level adaptation",
    "dividemix_cache": "DivideMix cache-level adaptation",
    "deft_cache": "DeFT cache-level adaptation (Wei et al., NeurIPS 2024)",
}


class LinearHead(nn.Module):
    def __init__(self, dim: int, classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_clip_prior(bundle: dict[str, Any], split: str, labels: torch.Tensor,
                  classes: int) -> torch.Tensor:
    prior = bundle.get(f"clip_probs_{split}")
    if isinstance(prior, torch.Tensor) and prior.ndim == 2 and prior.shape == (labels.shape[0], classes):
        return prior.float().clamp_min(1e-6).div(prior.float().sum(1, keepdim=True).clamp_min(1e-6))
    embeds = bundle.get("clip_label_embeds")
    if isinstance(embeds, torch.Tensor) and embeds.ndim == 2 and embeds.shape[0] == classes:
        features = bundle[f"{split}_feats"]
        logits = F.normalize(features.float(), dim=-1) @ F.normalize(embeds.float(), dim=-1).t()
        return logits.mul(20.0).softmax(dim=-1).cpu()
    return F.one_hot(labels.clamp(0, classes - 1), classes).float()


def fit_two_component_gmm(values: torch.Tensor, steps: int = 40) -> torch.Tensor:
    values = values.float().clamp(1e-5, 1.0)
    quantiles = torch.quantile(values, torch.tensor([0.25, 0.75]))
    means = quantiles.clone()
    variance = torch.full((2,), max(float(values.var()), 1e-3))
    mixture = torch.full((2,), 0.5)
    for _ in range(steps):
        log_prob = torch.log(mixture.clamp_min(1e-6))[None, :] - 0.5 * (
            (values[:, None] - means[None, :]).square() / variance[None, :]
            + torch.log(2.0 * torch.pi * variance)[None, :]
        )
        resp = log_prob.softmax(1)
        count = resp.sum(0).clamp_min(1e-6)
        mixture = count / count.sum()
        means = (resp * values[:, None]).sum(0) / count
        variance = (resp * (values[:, None] - means[None, :]).square()).sum(0) / count
        variance = variance.clamp_min(1e-4)
    return resp[:, means.argmin()]


def trim_pairwise_auc(logits: torch.Tensor, posterior: torch.Tensor,
                      keep_fraction: float, temperature: float,
                      max_pairs: int) -> torch.Tensor:
    """Compute per-sample weighted OVR pair losses and trim high-loss points."""
    if logits.shape[0] > max_pairs:
        selected = torch.randperm(logits.shape[0], device=logits.device)[:max_pairs]
        logits, posterior = logits[selected], posterior[selected]
    n, classes = logits.shape
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    sample_loss = torch.zeros(n, device=logits.device)
    for cls in range(classes):
        probability = posterior[:, cls].clamp(1e-5, 1.0 - 1e-5)
        difference = logits[:, cls, None] - logits[:, cls][None, :]
        positive_negative = probability[:, None] * (1.0 - probability)[None, :]
        negative_positive = (1.0 - probability)[:, None] * probability[None, :]
        pair_loss = (
            positive_negative * F.softplus(-difference / temperature)
            + negative_positive * F.softplus(difference / temperature)
        )
        pair_loss = pair_loss.masked_fill(eye, 0.0)
        sample_loss += pair_loss.sum(1) / max(1, n - 1)
    keep = max(1, min(n, int(round(n * keep_fraction))))
    return torch.topk(sample_loss, keep, largest=False).values.mean()


@torch.no_grad()
def per_sample_ce(model: nn.Module, features: torch.Tensor, labels: torch.Tensor,
                  device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    pieces = []
    for start in range(0, features.shape[0], batch_size):
        logits = model(features[start:start + batch_size].to(device))
        pieces.append(F.cross_entropy(logits, labels[start:start + batch_size].to(device), reduction="none").cpu())
    return torch.cat(pieces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(PAPER_INFO))
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--obs-labels", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--trim-keep-fraction", type=float, default=0.70)
    parser.add_argument("--pairwise-batch-size", type=int, default=256)
    parser.add_argument("--strong-noise", type=float, default=0.02)
    parser.add_argument("--unsup-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache_path = args.cache or str(clip_cache(args.dataset))
    obs_path = args.obs_labels or str(observed_labels_cache(args.dataset))
    run_dir = make_run_dir(args.dataset, args.method, args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_cache_and_observed(cache_path, obs_path)
    train_x, val_x, test_x = bundle["train_feats"], bundle["val_feats"], bundle["test_feats"]
    train_y, val_y, test_y = bundle["train_labels"], bundle["val_labels"], bundle["test_labels"]
    y_obs, verified = bundle["y_obs"], bundle["s"].eq(0)
    classes = int(bundle["num_classes"])
    clip_prior = get_clip_prior(bundle, "train", y_obs, classes)
    y_vlm = clip_prior.argmax(1)
    verified_count = int(verified.sum())
    rho = verified_count / max(1, len(verified))
    model = LinearHead(train_x.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history: list[dict[str, Any]] = []
    best_acc, best_epoch, bad_epochs = -1.0, 0, 0
    start_time = time.time()
    selection_info: dict[str, Any] = {"verified_count": verified_count, "rho": rho}

    # Fixed-label controls.
    if args.method == "fcl_filter_ce":
        targets = y_obs.clone()
        loader = DataLoader(TensorDataset(train_x, targets), batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)
    elif args.method in {"clip_conf_ce", "random_filter_ce"}:
        confidence = clip_prior.max(1).values
        if args.method == "clip_conf_ce":
            selected = torch.topk(confidence, verified_count, largest=False).indices
            selection_name = "lowest CLIP top-1 confidence"
        else:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            selected = torch.randperm(len(train_x), generator=generator)[:verified_count]
            selection_name = "uniform random selection"
        targets = y_vlm.clone()
        targets[selected] = train_y[selected]  # simulated correction of selected points only
        selection_info.update({
            "selection": selection_name,
            "selected_count": int(selected.numel()),
            "selected_vlm_accuracy": float(y_vlm[selected].eq(train_y[selected]).float().mean()),
        })
        loader = DataLoader(TensorDataset(train_x, targets), batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)
    elif args.method == "clip_zero_ce":
        targets = y_vlm.clone()
        selection_info.update({
            "selection": "none",
            "selected_count": 0,
            "selected_vlm_accuracy": None,
            "human_verification": False,
        })
        loader = DataLoader(TensorDataset(train_x, targets), batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)
    elif args.method == "deft_cache":
        top2 = torch.topk(clip_prior, 2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp(0, 1)
        clean_probability = fit_two_component_gmm(margin)
        selection_info.update({"detector": "CLIP class-vs-other margin", "clean_probability_mean": float(clean_probability.mean())})
        loader = DataLoader(TensorDataset(train_x, y_vlm, clean_probability),
                            batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    elif args.method == "dividemix_cache":
        loader = DataLoader(TensorDataset(train_x, y_vlm), batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)
    elif args.method in {"fixmatch_cache", "softmatch_cache"}:
        loader = DataLoader(TensorDataset(train_x, y_obs, verified), batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)
        confidence_mean, confidence_std = 0.8, 0.2
    elif args.method == "xie_trim_auc":
        loader = DataLoader(TensorDataset(train_x, clip_prior), batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        if args.method == "dividemix_cache":
            losses = per_sample_ce(model, train_x, y_vlm, device, args.batch_size)
            clean_probability = fit_two_component_gmm(losses / losses.max().clamp_min(1e-6))
            loader = DataLoader(TensorDataset(train_x, y_vlm, clean_probability),
                                batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for batch in loader:
            if args.method in {"fcl_filter_ce", "clip_conf_ce", "random_filter_ce", "clip_zero_ce"}:
                features, labels = batch
                logits = model(features.to(device).float())
                loss = F.cross_entropy(logits, labels.to(device))
            elif args.method == "deft_cache":
                features, labels, clean_probability = batch
                logits = model(features.to(device).float())
                weight = clean_probability.to(device).float().clamp(0.05, 1.0)
                loss = (F.cross_entropy(logits, labels.to(device), reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
            elif args.method == "dividemix_cache":
                features, labels, clean_probability = batch
                logits = model(features.to(device).float())
                weight = (0.2 + 0.8 * clean_probability).to(device).float()
                loss = (F.cross_entropy(logits, labels.to(device), reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
            elif args.method in {"fixmatch_cache", "softmatch_cache"}:
                features, labels, is_verified = batch
                features = features.to(device).float()
                labels, is_verified = labels.to(device).long(), is_verified.to(device).bool()
                weak_logits = model(features)
                labeled_loss = F.cross_entropy(weak_logits[is_verified], labels[is_verified]) if is_verified.any() else weak_logits.sum() * 0
                confidence, pseudo = weak_logits.softmax(-1).detach().max(-1)
                strong_features = F.normalize(features + args.strong_noise * torch.randn_like(features), dim=-1)
                strong_logits = model(strong_features)
                unsup_loss = F.cross_entropy(strong_logits[~is_verified], pseudo[~is_verified], reduction="none") if (~is_verified).any() else weak_logits.sum() * 0
                if args.method == "fixmatch_cache":
                    mask = (~is_verified) & (confidence >= args.confidence_threshold)
                    unsup = unsup_loss[confidence[~is_verified] >= args.confidence_threshold].mean() if mask.any() else weak_logits.sum() * 0
                else:
                    current_mean = float(confidence[~is_verified].mean().item()) if (~is_verified).any() else confidence_mean
                    confidence_mean = 0.99 * confidence_mean + 0.01 * current_mean
                    current_std = float(confidence[~is_verified].std().item()) if (~is_verified).sum() > 1 else confidence_std
                    confidence_std = 0.99 * confidence_std + 0.01 * max(0.05, current_std)
                    weights = torch.exp(-0.5 * ((confidence[~is_verified] - confidence_mean) / confidence_std).square())
                    unsup = (unsup_loss * weights).sum() / weights.sum().clamp_min(1.0) if unsup_loss.numel() else weak_logits.sum() * 0
                loss = labeled_loss + args.unsup_weight * unsup
            else:  # xie_trim_auc
                features, posterior = batch
                features, posterior = features.to(device).float(), posterior.to(device).float()
                logits = model(features)
                loss = trim_pairwise_auc(logits, posterior, args.trim_keep_fraction, 0.5, args.pairwise_batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())
            steps += 1
        scheduler.step()
        val_metrics = evaluate(model, val_x, val_y, device)
        history.append({"epoch": epoch, "train_loss": running / max(1, steps), **val_metrics})
        print(f"[{args.method}][{args.dataset}] epoch={epoch:03d} val_acc={val_metrics['accuracy'] * 100:.2f}%", flush=True)
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
    test_metrics = evaluate(model, test_x, test_y, device)
    reported_verified_count = 0 if args.method == "clip_zero_ce" else verified_count
    reported_rho = reported_verified_count / max(1, len(verified))
    selection_info.update({"method": args.method, "dataset": args.dataset,
                           "paper": PAPER_INFO[args.method], "cache_level": True,
                           "cache": cache_path, "observed_labels": obs_path,
                           "budget_verified_count": reported_verified_count,
                           "budget_rho": reported_rho,
                           "args": vars(args)})
    save_config(run_dir, selection_info)
    save_result(run_dir, {
        "method": args.method, "dataset": args.dataset, "paper": PAPER_INFO[args.method],
        "best_val_accuracy": best_acc, "best_epoch": best_epoch,
        "test": test_metrics, "verified_count": reported_verified_count, "rho": reported_rho,
        "epochs_completed": completed, "selection": selection_info,
        "elapsed_seconds": time.time() - start_time,
    })
    save_run_metadata(run_dir, method=args.method, dataset=args.dataset,
                      script="src/contrast/train_revision_baselines.py",
                      args=args, command=sys.argv,
                      extra={"cache_level_adaptation": True})
    print(f"[{args.method}][DONE] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
