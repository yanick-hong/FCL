"""Preview or execute dataset-specific corrected-label generation."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import clip_cache, observed_labels_cache


def main() -> None:
    registry = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(registry))
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--execute", action="store_true", help="Without this flag only print commands.")
    args = parser.parse_args()
    if not 0 < args.rho <= 1:
        parser.error("--rho must be in (0, 1]")
    commands = []
    for dataset in args.datasets:
        cache, observed = clip_cache(dataset), observed_labels_cache(dataset)
        if args.execute:
            if not cache.is_file():
                parser.error(f"Missing feature cache: {cache}")
            if observed.exists():
                parser.error(f"Refusing to overwrite existing labels: {observed}. Archive them first or reuse them.")
        command = [sys.executable, registry[dataset]["labels"], "--cache_path", str(cache),
                   "--save_path", str(observed), "--top_outlier_pct", str(args.rho)]
        commands.append(command)
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
