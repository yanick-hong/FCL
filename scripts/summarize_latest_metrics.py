"""Generate a method-dataset-accuracy table from the newest completed runs."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any


METHOD_ORDER = (
    "fcl_auc_ce", "nlprompt", "dcd", "ido", "dld",
    "clip_zero_ce", "random_filter_ce", "clip_conf_ce", "fcl_filter_ce",
    "xie_trim_auc", "fixmatch_cache", "softmatch_cache", "dividemix_cache",
    "deft_cache",
)
METHOD_NAMES = {
    "fcl_auc_ce": "FCL-AUC-CE", "nlprompt": "NLPrompt", "dcd": "DCD",
    "ido": "IDO", "dld": "DLD", "clip_zero_ce": "CLIP-Zero-CE",
    "random_filter_ce": "Random-Filter-CE", "clip_conf_ce": "CLIP-Conf-CE",
    "fcl_filter_ce": "FCL-Filter-CE", "xie_trim_auc": "Xie-Trim-AUC",
    "fixmatch_cache": "FixMatch*", "softmatch_cache": "SoftMatch*",
    "dividemix_cache": "DivideMix*", "deft_cache": "DeFT*",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_outputs = Path(os.environ.get("OUTPUT_ROOT", str(project_root / "outputs")))
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=default_outputs)
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def latest_metrics_file(method_dir: Path) -> Path | None:
    files = [path for path in method_dir.glob("*/metrics.json") if path.is_file()]
    return max(files, key=lambda path: (path.parent.name, path.stat().st_mtime_ns)) if files else None


def read_accuracy(metrics_file: Path) -> float | None:
    try:
        metrics: dict[str, Any] = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("best_val_accuracy", "best_val_acc"):
        if metrics.get(key) is not None:
            return float(metrics[key])
    for block_name in ("best", "validation", "val"):
        block = metrics.get(block_name)
        if isinstance(block, dict):
            for key in ("accuracy", "acc"):
                if block.get(key) is not None:
                    return float(block[key])
    return None


def collect_rows(outputs_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not outputs_root.is_dir():
        raise FileNotFoundError(outputs_root)
    for dataset_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        if dataset_dir.name.startswith("_") or dataset_dir.name in {"cache", "revision_summary"}:
            continue
        for method in METHOD_ORDER:
            metrics_file = latest_metrics_file(dataset_dir / method)
            if metrics_file is None:
                continue
            accuracy = read_accuracy(metrics_file)
            rows.append({
                "method": METHOD_NAMES[method],
                "dataset": dataset_dir.name,
                "accuracy": "-" if accuracy is None else f"{accuracy * 100:.2f}%",
            })
    return rows


def render_markdown(rows: list[dict[str, str]]) -> str:
    lines = ["| Method | Dataset | Accuracy |", "|---|---|---:|"]
    lines.extend(f"| {row['method']} | {row['dataset']} | {row['accuracy']} |" for row in rows)
    return "\n".join(lines) + "\n"


def render_csv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("method", "dataset", "accuracy"))
    for row in rows:
        writer.writerow((row["method"], row["dataset"], row["accuracy"]))
    return output.getvalue()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.outputs_root.expanduser().resolve())
    if not rows:
        raise SystemExit("No metrics.json files found.")
    content = render_markdown(rows) if args.format == "markdown" else render_csv(rows)
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
