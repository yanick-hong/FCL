"""Compare completed paper runs with the published reference table.

Only manifests written by ``run_paper_experiments.py`` are considered.  A
missing row is reported explicitly; reference percentages are never presented
as measured values.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "paper")
    parser.add_argument("--targets", type=Path, default=ROOT / "configs" / "paper_targets.yaml")
    parser.add_argument("--source", choices=("clip", "qwen"), default="clip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def metric_accuracy(metrics: dict) -> float | None:
    """Extract an accuracy in [0,1] from each supported result format."""
    if not isinstance(metrics, dict):
        return None
    test = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    nested = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else {}
    nested_test = nested.get("test") if isinstance(nested.get("test"), dict) else {}
    candidates = [metrics.get("test_accuracy"), test.get("accuracy"), nested_test.get("accuracy")]
    for value in candidates:
        if isinstance(value, bool):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0 <= value <= 100:
            return value / 100.0 if value > 1.0 else value
    return None


def completed_runs(root: Path, source: str, seed: int) -> dict[tuple[str, str], tuple[float, Path]]:
    """Select the newest completed main run for each dataset/method pair."""
    selected: dict[tuple[str, str], tuple[float, Path]] = {}
    for manifest in root.rglob("paper_run.json") if root.exists() else ():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "completed" or payload.get("suite") != "main":
            continue
        try:
            run_seed = int(payload.get("seed", -1))
        except (TypeError, ValueError):
            continue
        if payload.get("source") != source or run_seed != seed:
            continue
        metrics_path = manifest.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            accuracy = metric_accuracy(json.loads(metrics_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            accuracy = None
        if accuracy is None:
            continue
        stamp = manifest.stat().st_mtime
        key = (str(payload.get("dataset")), str(payload.get("method")))
        if key not in selected or stamp > selected[key][0]:
            selected[key] = (stamp, manifest.parent)
    return selected


def main() -> int:
    args = parse_args()
    target = load_yaml(args.targets)
    datasets = list(target["dataset_order"])
    references = target.get(args.source, {})
    methods = [method for method in references if isinstance(references[method], list)]
    runs = completed_runs(args.output_root, args.source, args.seed)
    rows = []
    for method in methods:
        values = references[method]
        for index, dataset in enumerate(datasets):
            manifest_dir = runs.get((dataset, method), (None, None))[1]
            measured = None
            if manifest_dir is not None:
                try:
                    measured = metric_accuracy(json.loads((manifest_dir / "metrics.json").read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    measured = None
            target_pct = float(values[index])
            rows.append({"source": args.source, "dataset": dataset, "method": method,
                         "seed": args.seed, "measured_pct": None if measured is None else 100 * measured,
                         "target_pct": target_pct,
                         "delta_pct_points": None if measured is None else 100 * measured - target_pct,
                         "status": "ok" if measured is not None else "missing",
                         "run_dir": None if manifest_dir is None else str(manifest_dir)})
    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"source={args.source} seed={args.seed}")
    print("dataset\tmethod\tmeasured(%)\ttarget(%)\tdelta(pp)\tstatus")
    for row in rows:
        measured = "missing" if row["measured_pct"] is None else f"{row['measured_pct']:.2f}"
        delta = "-" if row["delta_pct_points"] is None else f"{row['delta_pct_points']:+.2f}"
        print(f"{row['dataset']}\t{row['method']}\t{measured}\t{row['target_pct']:.2f}\t{delta}\t{row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
