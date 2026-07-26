"""Run and finalize the legacy FL-C/NL-C/L2B/DLD/CSGN baselines.

Each invocation gets an isolated ``outputs/<dataset>/<method>/<timestamp>``
directory.  The legacy training files remain usable as standalone scripts,
while this runner provides the common reproducibility and result format.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import OUTPUT_ROOT, PROCESSED_ROOT  # noqa: E402
from contrast.cache_experiment import make_run_dir, save_config, save_result, save_run_metadata  # noqa: E402


METHOD_SCRIPTS = {
    "fl_c": "src/contrast/train_trusted.py",
    "nl_c": "src/contrast/train_noise.py",
    "l2b": "src/contrast/train_l2b.py",
    "dld": "src/contrast/train_dld.py",
    "csgn": "src/contrast/train_csgn.py",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--methods", nargs="+", choices=sorted(METHOD_SCRIPTS),
                   default=list(METHOD_SCRIPTS))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def available_datasets() -> list[str]:
    suffix = "_clip_vit-l-14_embeddings.pt"
    return sorted(p.name[:-len(suffix)] for p in PROCESSED_ROOT.glob(f"*{suffix}"))


def command_for(method: str, dataset: str, run_dir: Path, cache: Path,
                observed: Path, args: argparse.Namespace) -> list[str]:
    script = METHOD_SCRIPTS[method]
    common = [sys.executable, script]
    if method == "fl_c":
        return common + ["--cache", str(cache), "--save", str(run_dir / "best.ckpt"),
                          "--epochs", str(args.epochs), "--patience", str(args.patience),
                          "--batch_size", str(args.batch_size)]
    if method == "nl_c":
        return common + ["--cache", str(cache), "--obs_labels_path", str(observed),
                          "--max_epochs", str(args.epochs), "--patience", str(args.patience),
                          "--P", str(args.P), "--K", str(args.K),
                          "--best_ckpt", str(run_dir / "best.ckpt"),
                          "--val_dir", str(run_dir / "logs"), "--seed", str(args.seed),
                          "--use_ema"]
    if method == "l2b":
        return common + ["--cache", str(cache), "--obs_labels_path", str(observed),
                          "--max_epochs", str(args.epochs), "--patience", str(args.patience),
                          "--P", str(args.P), "--K", str(args.K),
                          "--best_ckpt", str(run_dir / "best.ckpt"),
                          "--val_dir", str(run_dir / "logs"), "--seed", str(args.seed)]
    if method == "dld":
        return common + ["--cache", str(cache), "--obs_labels_path", str(observed),
                          "--max_epochs", str(args.epochs), "--patience", str(args.patience),
                          "--P", str(args.P), "--K", str(args.K),
                          "--best_ckpt", str(run_dir / "best.ckpt"),
                          "--val_dir", str(run_dir / "logs"), "--seed", str(args.seed)]
    return common + ["--cache_path", str(cache), "--obs_labels_path", str(observed),
                     "--epochs", str(args.epochs), "--patience", str(args.patience),
                     "--P", str(args.P), "--K", str(args.K),
                     "--best_ckpt", str(run_dir / "best.ckpt"),
                     "--val_dir", str(run_dir / "logs"), "--seed", str(args.seed)]


def read_history(run_dir: Path) -> list[dict[str, str]]:
    candidates = [run_dir / "logs" / "val_metrics.csv",
                  run_dir / "logs" / "val_metrics_dld_linear.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            try:
                value = float(row[key])
                # Older CSGN versions wrote percentages.  New runs write [0,1].
                return value / 100.0 if key == "val_acc" and value > 1.0 else value
            except ValueError:
                pass
    return None


def best_from_history(history: list[dict[str, str]]) -> tuple[float | None, int | None]:
    values = [(as_float(row, "val_acc", "val_acc_head"), row) for row in history]
    values = [(v, row) for v, row in values if v is not None]
    if not values:
        return None, None
    value, row = max(values, key=lambda item: item[0])
    try:
        epoch = int(float(row.get("epoch", "-1")))
    except ValueError:
        epoch = None
    return value, epoch


def load_linear_state(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    pairs = [("linear.weight", "linear.bias"), ("fc.weight", "fc.bias"),
             ("classifier.weight", "classifier.bias")]
    for wk, bk in pairs:
        if wk in state and bk in state:
            return state[wk].float(), state[bk].float()
    raise KeyError(f"No linear classifier weights found in {path}; keys={list(state)[:10]}")


@torch.no_grad()
def test_metrics(cache_path: Path, ckpt_path: Path) -> dict[str, object]:
    cache = torch.load(cache_path, map_location="cpu")
    labels = cache.get("test_labels")
    feats = cache.get("test_feats")
    if not isinstance(feats, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return {"accuracy": None, "valid_count": 0, "total_count": 0,
                "note": "test split is not present in cache"}
    weight, bias = load_linear_state(ckpt_path)
    feats = F.normalize(feats.float(), dim=-1)
    labels = labels.long().view(-1)
    valid = (labels >= 0) & (labels < weight.shape[0])
    if not valid.any():
        return {"accuracy": None, "valid_count": 0, "total_count": int(labels.numel()),
                "note": "test labels are unavailable or invalid"}
    logits = feats[valid] @ weight.t() + bias
    y = labels[valid]
    return {
        "accuracy": float((logits.argmax(-1) == y).float().mean().item()),
        "nll": float(F.cross_entropy(logits, y).item()),
        "valid_count": int(valid.sum().item()),
        "total_count": int(labels.numel()),
    }


def finalize(run_dir: Path, method: str, dataset: str, cache: Path,
             command: list[str], args: argparse.Namespace, start: str,
             status: str, error: str | None = None) -> None:
    history = read_history(run_dir)
    best_acc, best_epoch = best_from_history(history)
    test = test_metrics(cache, run_dir / "best.ckpt") if (run_dir / "best.ckpt").exists() else {
        "accuracy": None, "valid_count": 0, "total_count": 0,
        "note": "best checkpoint was not produced"
    }
    payload = {
        "method": method,
        "dataset": dataset,
        "selection": {"metric": "val_accuracy", "best_val_accuracy": best_acc,
                       "best_epoch": best_epoch, "epochs_completed": len(history),
                       "history": "logs/val_metrics.csv"},
        "test": test,
        "status": status,
    }
    if error:
        payload["error"] = error
    save_result(run_dir, payload)
    save_config(run_dir, {"method": method, "dataset": dataset,
                          "script": METHOD_SCRIPTS[method], "command": command,
                          "args": vars(args), "cache": str(cache)})
    save_run_metadata(run_dir, method=method, dataset=dataset,
                      script=METHOD_SCRIPTS[method], args=args,
                      command=command, start_time=start, end_time=now(),
                      extra={"status": status, "error": error})


def main() -> None:
    args = parse_args()
    datasets = args.datasets or available_datasets()
    for dataset in datasets:
        cache = PROCESSED_ROOT / f"{dataset}_clip_vit-l-14_embeddings.pt"
        observed = PROCESSED_ROOT / f"{dataset}_observed_labels.pt"
        if not cache.exists() or not observed.exists():
            print(f"[SKIP] {dataset}: missing cache or observed labels", flush=True)
            continue
        for method in args.methods:
            run_dir = make_run_dir(dataset, method)
            command = command_for(method, dataset, run_dir, cache, observed, args)
            start = now()
            print("[RUN]", " ".join(command), flush=True)
            status, error = "completed", None
            try:
                subprocess.run(command, cwd=ROOT, check=True)
            except subprocess.CalledProcessError as exc:
                status, error = "failed", f"returncode={exc.returncode}"
                print(f"[ERROR] {dataset}/{method}: {error}", flush=True)
            except OSError as exc:
                status, error = "failed", repr(exc)
                print(f"[ERROR] {dataset}/{method}: {error}", flush=True)
            finalize(run_dir, method, dataset, cache, command, args, start, status, error)
            if status == "failed" and not args.force:
                continue


if __name__ == "__main__":
    main()
