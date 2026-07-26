"""Run validation-only sensitivity experiments for FCL's posterior mixing mu."""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cifar100", "food101", "tinyimagenet200"])
    parser.add_argument("--mus", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    processed = root / "data" / "processed"
    for dataset in args.datasets:
        cache = processed / f"{dataset}_clip_vit-l-14_embeddings.pt"
        observed = processed / f"{dataset}_observed_labels.pt"
        if not cache.exists() or not observed.exists():
            print(f"[SKIP] {dataset}: cache or observed labels missing", flush=True)
            continue
        for mu in args.mus:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{dataset}/fcl_mu_sensitivity/mu_{mu:.2f}/{stamp}"
            run_dir = OUTPUT_ROOT / name
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "src/fcl/train_auc_ce.py",
                       "--cache", str(cache), "--obs_labels_path", str(observed),
                       "--mu", str(mu), "--max_epochs", str(args.epochs),
                       "--seed", str(args.seed), "--best_ckpt", str(run_dir / "best.ckpt"),
                       "--val_dir", str(run_dir / "logs"), "--experiment_name", name]
            print("[RUN]", " ".join(command), flush=True)
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
