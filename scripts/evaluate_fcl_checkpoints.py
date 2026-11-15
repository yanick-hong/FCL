"""Evaluate existing FCL best checkpoints on the cached test splits.

This repairs the metrics contract of legacy FCL runs without retraining them.
Only the test fields are added; best validation fields and checkpoints remain
unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paths import clip_cache, OUTPUT_ROOT


def latest_metrics(output_root: Path, dataset: str) -> Path | None:
    files = sorted((output_root / dataset / "fcl_auc_ce").glob("*/metrics.json"))
    return files[-1] if files else None


def evaluate_checkpoint(checkpoint: Path, cache_path: Path) -> dict[str, float]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    features = F.normalize(cache["test_feats"].float(), dim=-1)
    labels = cache["test_labels"].long()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "fc.weight" in state:
        state = {key.removeprefix("fc."): value for key, value in state.items()}
    # A split may omit one or more classes.  The trained head defines the full
    # vocabulary; inferring it from test/train label maxima can truncate it and
    # fail to load an otherwise valid checkpoint.
    if not isinstance(state, dict) or not isinstance(state.get("weight"), torch.Tensor):
        raise ValueError(f"Checkpoint has no linear weight matrix: {checkpoint}")
    classes = int(state["weight"].shape[0])
    model = nn.Linear(features.shape[1], classes)
    model.load_state_dict(state)
    with torch.no_grad():
        logits = model(features)
        valid = (labels >= 0) & (labels < classes)
        if not valid.any():
            return {"loss": None, "accuracy": None, "valid_count": 0,
                    "total_count": int(labels.numel()),
                    "note": "test labels unavailable; use validation accuracy"}
        loss = F.cross_entropy(logits[valid], labels[valid]).item()
        accuracy = logits[valid].argmax(1).eq(labels[valid]).float().mean().item()
    return {"loss": float(loss), "accuracy": float(accuracy),
            "valid_count": int(valid.sum().item()), "total_count": int(labels.numel())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()
    output_root = args.outputs_root.expanduser().resolve()
    datasets = args.datasets or sorted(
        path.name for path in output_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
        and (path / "fcl_auc_ce").is_dir()
    )
    updated = 0
    for dataset in datasets:
        metrics_path = latest_metrics(output_root, dataset)
        cache_path = clip_cache(dataset)
        if metrics_path is None or not cache_path.is_file():
            print(f"[SKIP] {dataset}: metrics or cache missing")
            continue
        checkpoint = metrics_path.parent / "best.ckpt"
        if not checkpoint.is_file():
            print(f"[SKIP] {dataset}: {checkpoint} missing")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["test"] = evaluate_checkpoint(checkpoint, cache_path)
        metrics["test_loss"] = metrics["test"]["loss"]
        metrics["test_accuracy"] = metrics["test"]["accuracy"]
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        if metrics["test_accuracy"] is None:
            print(f"[UPDATED] {dataset}: test labels unavailable; validation metric retained -> {metrics_path}")
        else:
            print(f"[UPDATED] {dataset}: test_acc={metrics['test_accuracy'] * 100:.2f}% -> {metrics_path}")
        updated += 1
    print(f"[DONE] Updated {updated} FCL runs")


if __name__ == "__main__":
    main()
