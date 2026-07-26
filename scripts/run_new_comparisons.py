"""Run the new NeurIPS 2025 IDO and CVPR 2025 DLD cache adaptations."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--ido-stage1-epochs", type=int, default=20)
    parser.add_argument("--ido-stage2-epochs", type=int, default=100)
    parser.add_argument("--dld-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def latest_run(root: Path, dataset: str, method: str) -> Path | None:
    runs = sorted((root / "outputs" / dataset / method).glob("*/metrics.json"))
    return runs[-1].parent if runs else None


def run(command: list[str], root: Path) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    available = sorted(
        path.name.removesuffix("_clip_vit-l-14_embeddings.pt")
        for path in processed.glob("*_clip_vit-l-14_embeddings.pt")
    )
    datasets = args.datasets or available
    for dataset in datasets:
        cache = processed / f"{dataset}_clip_vit-l-14_embeddings.pt"
        observed = processed / f"{dataset}_observed_labels.pt"
        if not cache.exists() or not observed.exists():
            print(f"[SKIP] {dataset}: cache or observed labels missing", flush=True)
            continue
        if latest_run(root, dataset, "ido") is None:
            run([sys.executable, "src/contrast/train_ido_cache.py",
                 "--dataset", dataset, "--cache", str(cache),
                 "--obs-labels", str(observed),
                 "--stage1-epochs", str(args.ido_stage1_epochs),
                 "--stage2-epochs", str(args.ido_stage2_epochs),
                 "--batch-size", str(args.batch_size)], root)
        else:
            print(f"[SKIP] {dataset}/ido already has a completed run", flush=True)
        if latest_run(root, dataset, "dld") is None:
            run([sys.executable, "src/contrast/train_dld_cache.py",
                 "--dataset", dataset, "--cache", str(cache),
                 "--obs-labels", str(observed), "--epochs", str(args.dld_epochs),
                 "--batch-size", str(args.batch_size)], root)
        else:
            print(f"[SKIP] {dataset}/dld already has a completed run", flush=True)


if __name__ == "__main__":
    main()
