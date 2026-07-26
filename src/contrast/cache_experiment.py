"""Shared utilities for cache-only noisy-label experiments."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from paths import OUTPUT_ROOT


def make_run_dir(dataset: str, method: str, run_dir: str | None = None) -> Path:
    if run_dir:
        path = Path(run_dir).expanduser()
        if not path.is_absolute():
            # Accept both ``dataset/method/run`` and ``outputs/dataset/method/run``.
            path = ((OUTPUT_ROOT.parent / path) if path.parts and path.parts[0] == OUTPUT_ROOT.name
                    else (OUTPUT_ROOT / path)).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_ROOT / dataset / method / stamp
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def load_cache_and_observed(cache_path: str, obs_path: str) -> Dict[str, torch.Tensor | list]:
    cache = torch.load(cache_path, map_location="cpu")
    obs = torch.load(obs_path, map_location="cpu")
    required = ("train_feats", "val_feats", "test_feats", "train_labels",
                "val_labels", "test_labels")
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(f"Cache {cache_path} is missing: {missing}")
    if "y_obs" not in obs or "s" not in obs:
        raise KeyError(f"Observed-label cache {obs_path} must contain y_obs and s")
    n = cache["train_feats"].shape[0]
    if obs["y_obs"].numel() != n or obs["s"].numel() != n:
        raise ValueError(f"Observed labels length does not match train cache: {n}")
    result: Dict[str, Any] = dict(cache)
    result["y_obs"] = obs["y_obs"].long()
    result["s"] = obs["s"].long()
    for key in ("train_feats", "val_feats", "test_feats"):
        result[key] = F.normalize(result[key].float(), dim=-1)
    for key in ("train_labels", "val_labels", "test_labels"):
        result[key] = result[key].long()
    result["num_classes"] = int(
        cache.get("clip_label_embeds", cache["train_labels"]).shape[0]
    )
    return result


@torch.no_grad()
def classification_metrics(logits: torch.Tensor, labels: torch.Tensor,
                            ece_bins: int = 15) -> Dict[str, float]:
    labels = labels.long()
    probs = logits.float().softmax(dim=-1)
    pred = probs.argmax(dim=-1)
    correct = pred.eq(labels)
    n = max(1, labels.numel())
    metrics: Dict[str, float] = {
        "accuracy": float(correct.float().mean().item()),
        "nll": float(F.cross_entropy(logits.float(), labels).item()),
    }
    classes = int(logits.shape[-1])
    f1_values = []
    for c in range(classes):
        tp = ((pred == c) & (labels == c)).sum().float()
        fp = ((pred == c) & (labels != c)).sum().float()
        fn = ((pred != c) & (labels == c)).sum().float()
        denom = 2 * tp + fp + fn
        f1_values.append(float((2 * tp / denom).item()) if denom.item() else 0.0)
    metrics["macro_f1"] = sum(f1_values) / max(1, len(f1_values))
    confidence, _ = probs.max(dim=-1)
    ece = torch.zeros((), dtype=torch.float32)
    for b in range(ece_bins):
        lo, hi = b / ece_bins, (b + 1) / ece_bins
        mask = (confidence > lo) & (confidence <= hi if b else confidence <= hi)
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    metrics["ece"] = float(ece.item())
    return metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, feats: torch.Tensor, labels: torch.Tensor,
             device: torch.device, batch_size: int = 4096) -> Dict[str, float]:
    model.eval()
    pieces = []
    for start in range(0, feats.shape[0], batch_size):
        pieces.append(model(feats[start:start + batch_size].to(device)))
    return classification_metrics(torch.cat(pieces).cpu(), labels.cpu())


def save_history(run_dir: Path, history: Iterable[Dict[str, Any]]) -> None:
    rows = list(history)
    if not rows:
        return
    with (run_dir / "logs" / "val_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_result(run_dir: Path, payload: Dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    (run_dir / "metrics.json").write_text(
        json.dumps(convert(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_config(run_dir: Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    except ImportError:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    (run_dir / "config.yaml").write_text(text, encoding="utf-8")
