"""Run revision baselines with reproducible single- or multi-seed protocols."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METHODS = (
    "clip_conf_ce", "random_filter_ce", "clip_zero_ce", "fcl_filter_ce",
    "xie_trim_auc", "fixmatch_cache", "softmatch_cache", "dividemix_cache",
    "deft_cache",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Optional repeated seeds; overrides --seed.")
    parser.add_argument("--force", action="store_true",
                        help="Run even when a completed result for the seed exists.")
    return parser.parse_args()


def has_completed_seed(root: Path, dataset: str, method: str, seed: int) -> bool:
    for metrics_file in (root / "outputs" / dataset / method).glob("*/metrics.json"):
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            recorded = metrics.get("selection", {}).get("args", {}).get("seed")
            if recorded is not None and int(recorded) == int(seed):
                return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return False


def run(command: list[str], root: Path) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    available = sorted(path.name.removesuffix("_clip_vit-l-14_embeddings.pt")
                       for path in processed.glob("*_clip_vit-l-14_embeddings.pt"))
    datasets = args.datasets or available
    seeds = args.seeds or [args.seed]
    for dataset in datasets:
        cache = processed / f"{dataset}_clip_vit-l-14_embeddings.pt"
        observed = processed / f"{dataset}_observed_labels.pt"
        if not cache.exists() or not observed.exists():
            print(f"[SKIP] {dataset}: cache or observed labels missing", flush=True)
            continue
        for method in args.methods:
            for seed in seeds:
                if not args.force and has_completed_seed(root, dataset, method, seed):
                    print(f"[SKIP] {dataset}/{method}/seed={seed} already completed", flush=True)
                    continue
                run([sys.executable, "src/contrast/train_revision_baselines.py",
                     "--method", method, "--dataset", dataset,
                     "--cache", str(cache), "--obs-labels", str(observed),
                     "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                     "--seed", str(seed)], root)


if __name__ == "__main__":
    main()
