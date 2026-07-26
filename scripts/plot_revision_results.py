"""Summarize revision experiments and create publication-ready plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = (
    "fcl_auc_ce", "clip_zero_ce", "random_filter_ce", "clip_conf_ce",
    "fcl_filter_ce", "xie_trim_auc", "fixmatch_cache", "softmatch_cache",
    "dividemix_cache", "deft_cache",
)
DISPLAY = {
    "fcl_auc_ce": "FCL", "clip_zero_ce": "CLIP-Zero-CE",
    "random_filter_ce": "Random-Filter-CE", "clip_conf_ce": "CLIP-Conf-CE",
    "fcl_filter_ce": "FCL-Filter-CE", "xie_trim_auc": "Xie-Trim-AUC",
    "fixmatch_cache": "FixMatch*", "softmatch_cache": "SoftMatch*",
    "dividemix_cache": "DivideMix*", "deft_cache": "DeFT*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def latest(dataset: Path, method: str) -> Path | None:
    files = sorted((dataset / method).glob("*/metrics.json"))
    return files[-1] if files else None


def val_accuracy(metrics: dict) -> float | None:
    value = metrics.get("best_val_accuracy", metrics.get("best_val_acc"))
    return None if value is None else float(value)


def test_accuracy(metrics: dict) -> float | None:
    block = metrics.get("test")
    if isinstance(block, dict) and block.get("accuracy") is not None:
        return float(block["accuracy"])
    value = metrics.get("test_accuracy")
    return None if value is None else float(value)


def infer_verified_budget(metrics: dict, dataset: str, method: str, outputs_root: Path) -> tuple[int | None, float | None]:
    verified = metrics.get("verified_count")
    rho = metrics.get("rho")
    if verified is not None:
        return int(verified), None if rho is None else float(rho)
    if method != "fcl_auc_ce":
        return None, None
    # The legacy FCL trainer predates the standardized metrics contract.  Its
    # observed-label cache still contains the exact simulated verification mask.
    cache_path = outputs_root.parent / "data" / "processed" / f"{dataset}_observed_labels.pt"
    try:
        import torch
        observed = torch.load(cache_path, map_location="cpu")
        mask = observed["s"].eq(0)
        count = int(mask.sum().item())
        return count, count / max(1, int(mask.numel()))
    except (OSError, KeyError, TypeError, RuntimeError, ImportError):
        return None, None


def main() -> None:
    args = parse_args()
    root = (args.outputs_root or (Path(__file__).resolve().parents[1] / "outputs")).resolve()
    output = (args.output_dir or root / "revision_summary").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        if dataset.name in {"cache", "revision_summary", "mu_sensitivity_summary"}:
            continue
        for method in METHODS:
            file = latest(dataset, method)
            if file is None:
                continue
            try:
                metrics = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[WARN] Cannot read {file}: {exc}")
                continue
            verified, rho = infer_verified_budget(metrics, dataset.name, method, root)
            rows.append({
                "method": DISPLAY[method],
                "dataset": dataset.name,
                "accuracy": val_accuracy(metrics),
                "test_accuracy": test_accuracy(metrics),
                "rho": rho,
                "verified_count": verified,
                "human_cost_usd": None if verified is None else float(verified) * 0.04,
                "path": str(file),
            })
    rows.sort(key=lambda row: (row["dataset"], row["method"]))
    fields = ("method", "dataset", "accuracy", "test_accuracy", "rho",
              "verified_count", "human_cost_usd", "path")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Method | Dataset | Best validation accuracy | Test accuracy | Human cost (USD) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        val = "-" if row["accuracy"] is None else f"{row['accuracy'] * 100:.2f}%"
        test = "-" if row["test_accuracy"] is None else f"{row['test_accuracy'] * 100:.2f}%"
        cost = "-" if row["human_cost_usd"] is None else f"${row['human_cost_usd']:.2f}"
        lines.append(f"| {row['method']} | {row['dataset']} | {val} | {test} | {cost} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[DONE] {len(rows)} rows; matplotlib unavailable")
        return

    datasets = sorted({row["dataset"] for row in rows})
    methods = [method for method in METHODS if any(row["method"] == DISPLAY[method] for row in rows)]
    x = list(range(len(datasets)))
    width = 0.82 / max(1, len(methods))
    plt.figure(figsize=(max(12, len(datasets) * 2), 6))
    for index, method in enumerate(methods):
        values = []
        for dataset in datasets:
            match = next((row for row in rows if row["dataset"] == dataset
                          and row["method"] == DISPLAY[method]), None)
            values.append(float("nan") if match is None or match["accuracy"] is None
                          else match["accuracy"] * 100)
        offsets = [value + (index - (len(methods) - 1) / 2) * width for value in x]
        plt.bar(offsets, values, width=width, label=DISPLAY[method])
    plt.xticks(x, datasets, rotation=25, ha="right")
    plt.ylabel("Best validation accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "best_validation_accuracy.png", dpi=220)
    plt.close()

    budget_methods = {"FCL", "CLIP-Zero-CE", "Random-Filter-CE", "CLIP-Conf-CE", "FCL-Filter-CE"}
    plt.figure(figsize=(9, 6))
    for method in methods:
        display = DISPLAY[method]
        if display not in budget_methods:
            continue
        subset = [row for row in rows if row["method"] == display
                  and row["human_cost_usd"] is not None and row["accuracy"] is not None]
        if subset:
            plt.scatter([row["human_cost_usd"] for row in subset],
                        [row["accuracy"] * 100 for row in subset], s=55, label=display)
    plt.xlabel("Human verification cost (USD; $0.04/image)")
    plt.ylabel("Best validation accuracy (%)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "cost_accuracy_tradeoff.png", dpi=220)
    plt.close()
    print(f"[DONE] {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
