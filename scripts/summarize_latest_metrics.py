"""Summarize the newest completed run for every method and dataset.

The script reads ``outputs/<dataset>/<method>/<timestamp>/metrics.json`` and
prints a Markdown comparison table.  FCL runs created by older code use
``best_val_acc`` while newer runs use ``best_val_accuracy``; both are
supported.

Examples
--------
    python scripts/summarize_latest_metrics.py
    python scripts/summarize_latest_metrics.py --output outputs/latest_comparison.md
    python scripts/summarize_latest_metrics.py --format csv --output outputs/latest_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


METHOD_ORDER = ("fcl_auc_ce", "nlprompt", "dcd", "ido", "dld")
METHOD_NAMES = {
    "fcl_auc_ce": "FCL-AUC-CE",
    "nlprompt": "NLPrompt",
    "dcd": "DCD",
    "ido": "IDO",
    "dld": "DLD",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_outputs = Path(
        os.environ.get("OUTPUT_ROOT", str(project_root / "outputs"))
    )
    parser = argparse.ArgumentParser(
        description="Generate a method-dataset-accuracy table from latest metrics."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_outputs,
        help="Root of the outputs directory (default: $OUTPUT_ROOT or ./outputs).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Table format (default: markdown).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Without it, the table is printed to stdout.",
    )
    return parser.parse_args()


def latest_metrics_file(method_dir: Path) -> Path | None:
    files = [path for path in method_dir.glob("*/metrics.json") if path.is_file()]
    if not files:
        return None
    # Experiment timestamps are lexicographically sortable.  The mtime is a
    # deterministic tie-breaker for manually copied runs with the same name.
    return max(files, key=lambda path: (path.parent.name, path.stat().st_mtime_ns))


def read_accuracy(metrics_file: Path) -> float | None:
    try:
        metrics: dict[str, Any] = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Cannot read {metrics_file}: {exc}", file=sys.stderr)
        return None

    # ``best_val_acc`` is the field used by the original FCL trainer.
    for key in ("best_val_accuracy", "best_val_acc"):
        value = metrics.get(key)
        if value is not None:
            return float(value)

    # Also accept a nested validation block for future trainers.
    for block_name in ("best", "validation", "val"):
        block = metrics.get(block_name)
        if isinstance(block, dict):
            for key in ("accuracy", "acc"):
                value = block.get(key)
                if value is not None:
                    return float(value)
    return None


def collect_rows(outputs_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_root}")

    for dataset_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        if dataset_dir.name.startswith("_") or dataset_dir.name == "cache":
            continue
        for method in METHOD_ORDER:
            metrics_file = latest_metrics_file(dataset_dir / method)
            if metrics_file is None:
                continue
            accuracy = read_accuracy(metrics_file)
            rows.append(
                {
                    "method": METHOD_NAMES[method],
                    "dataset": dataset_dir.name,
                    "accuracy": "—" if accuracy is None else f"{accuracy * 100:.2f}%",
                }
            )
    return rows


def render_markdown(rows: list[dict[str, str]]) -> str:
    lines = ["| 方法 | 数据集 | 精度 |", "|---|---|---:|"]
    lines.extend(
        f"| {row['method']} | {row['dataset']} | {row['accuracy']} |" for row in rows
    )
    return "\n".join(lines) + "\n"


def render_csv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("方法", "数据集", "精度"))
    for row in rows:
        writer.writerow((row["method"], row["dataset"], row["accuracy"]))
    return output.getvalue()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.outputs_root.expanduser().resolve())
    if not rows:
        print("No metrics.json files found.", file=sys.stderr)
        return
    content = render_markdown(rows) if args.format == "markdown" else render_csv(rows)
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
