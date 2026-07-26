"""Run FCL, NLPrompt and DCD with a stable output hierarchy."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--fcl-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--skip-fcl", action="store_true")
    return parser.parse_args()


def slug_from_cache(path: Path) -> str:
    return path.name.removesuffix("_clip_vit-l-14_embeddings.pt")


def latest_run(root: Path, dataset: str, method: str) -> Path | None:
    runs = sorted((root / "outputs" / dataset / method).glob("*/metrics.json"))
    return runs[-1].parent if runs else None


def run(cmd: list[str], root: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    available = sorted(slug_from_cache(p) for p in processed.glob("*_clip_vit-l-14_embeddings.pt"))
    datasets = args.datasets or available
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for dataset in datasets:
        cache = processed / f"{dataset}_clip_vit-l-14_embeddings.pt"
        obs = processed / f"{dataset}_observed_labels.pt"
        if not cache.exists() or not obs.exists():
            print(f"[SKIP] {dataset}: cache or observed labels missing", flush=True)
            continue
        if not args.skip_fcl and latest_run(root, dataset, "fcl_auc_ce") is None:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = root / "outputs" / dataset / "fcl_auc_ce" / stamp
            run_dir.mkdir(parents=True, exist_ok=True)
            run([sys.executable, "src/fcl/train_auc_ce.py", "--cache", str(cache),
                 "--obs_labels_path", str(obs), "--max_epochs", str(args.fcl_epochs),
                 "--best_ckpt", str(run_dir / "best.ckpt"),
                 "--val_dir", str(run_dir / "logs"),
                 "--experiment_name", f"{dataset}/fcl_auc_ce/{stamp}"], root)
        if latest_run(root, dataset, "nlprompt") is None:
            run([sys.executable, "src/contrast/train_nlprompt.py", "--dataset", dataset,
                 "--cache", str(cache), "--obs-labels", str(obs), "--epochs", str(args.epochs),
                 "--batch-size", str(args.batch_size)], root)
        else:
            print(f"[SKIP] {dataset}/nlprompt already has a completed run", flush=True)
        if latest_run(root, dataset, "dcd") is None:
            run([sys.executable, "src/contrast/train_dcd.py", "--dataset", dataset,
                 "--cache", str(cache), "--obs-labels", str(obs), "--epochs", str(args.epochs),
                 "--batch-size", str(args.batch_size)], root)
        else:
            print(f"[SKIP] {dataset}/dcd already has a completed run", flush=True)


if __name__ == "__main__":
    main()
