"""Plot FCL posterior-mixing sensitivity and training stability.

Expected layout::

    outputs/<dataset>/fcl_mu_sensitivity/mu_<value>/<timestamp>/metrics.json
    outputs/<dataset>/fcl_mu_sensitivity/mu_<value>/<timestamp>/logs/val_metrics.csv

The script writes a CSV/Markdown summary, a mu-versus-accuracy plot, and
per-dataset validation curves. It tolerates incomplete runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


MU_RE = re.compile(r"^mu_(?P<mu>[-+]?\d+(?:\.\d+)?)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def newest_metrics(mu_dir: Path) -> Path | None:
    files = sorted(mu_dir.glob("*/metrics.json"))
    return files[-1] if files else None


def read_curve(path: Path) -> list[tuple[int, float]]:
    if not path.is_file():
        return []
    points: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                points.append((int(row["epoch"]), float(row["val_acc"])))
            except (KeyError, TypeError, ValueError):
                continue
    return points


def main() -> None:
    args = parse_args()
    root = (args.outputs_root or Path(__file__).resolve().parents[1] / "outputs").resolve()
    output = (args.output_dir or root / "mu_sensitivity_summary").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    curves: dict[str, dict[float, list[tuple[int, float]]]] = {}
    for dataset_dir in sorted(root.iterdir() if root.is_dir() else []):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("_"):
            continue
        method_dir = dataset_dir / "fcl_mu_sensitivity"
        if not method_dir.is_dir():
            continue
        for mu_dir in sorted(method_dir.iterdir()):
            match = MU_RE.match(mu_dir.name)
            if not match or not mu_dir.is_dir():
                continue
            metrics_path = newest_metrics(mu_dir)
            if metrics_path is None:
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                best = float(metrics["best_val_accuracy"])
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            mu = float(match.group("mu"))
            curve = read_curve(metrics_path.parent / "logs" / "val_metrics.csv")
            curves.setdefault(dataset_dir.name, {})[mu] = curve
            rows.append({
                "dataset": dataset_dir.name,
                "mu": mu,
                "best_val_accuracy": best,
                "best_epoch": metrics.get("best_epoch", ""),
                "epochs_completed": metrics.get("epochs_completed", ""),
                "path": str(metrics_path),
            })

    rows.sort(key=lambda row: (str(row["dataset"]), float(row["mu"])))
    fields = ("dataset", "mu", "best_val_accuracy", "best_epoch", "epochs_completed", "path")
    with (output / "mu_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Dataset | μ | Best validation accuracy | Best epoch |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {float(row['mu']):.2f} | {float(row['best_val_accuracy']) * 100:.2f}% | {row['best_epoch']} |")
    (output / "mu_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[DONE] {len(rows)} rows; matplotlib unavailable")
        return

    datasets = sorted(curves)
    if rows:
        plt.figure(figsize=(8, 5))
        for dataset in datasets:
            subset = sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: float(row["mu"]))
            plt.plot([float(row["mu"]) for row in subset], [float(row["best_val_accuracy"]) * 100 for row in subset], marker="o", label=dataset)
        plt.xlabel("μ (model posterior weight)")
        plt.ylabel("Best validation accuracy (%)")
        plt.xticks(sorted({float(row["mu"]) for row in rows}))
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "mu_vs_accuracy.png", dpi=220)
        plt.close()

    for dataset, dataset_curves in curves.items():
        plt.figure(figsize=(8, 5))
        for mu, curve in sorted(dataset_curves.items()):
            if curve:
                plt.plot([epoch for epoch, _ in curve], [acc * 100 for _, acc in curve], label=f"μ={mu:.2f}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation accuracy (%)")
        plt.title(f"FCL training stability: {dataset}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{dataset}_training_stability.png", dpi=220)
        plt.close()
    print(f"[DONE] {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
