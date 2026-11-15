"""Summarize completed, independent training seeds and matched objective contrasts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


IDENTITY = ("dataset", "label_source", "selection", "objective")
CONSISTENT = ("cache_sha256", "obs_sha256", "mask_sha256", "y_obs_sha256",
              "sample_order_sha256", "source_sha256", "protocol_sha256")


def mean_interval(values):
    """Sample SD and two-sided Student-t 95% CI, in the input units."""
    from scipy.stats import t
    if len(values) < 3 or any(not math.isfinite(x) for x in values):
        raise ValueError("At least three finite independent seed results are required")
    avg, sd = statistics.mean(values), statistics.stdev(values)
    half = float(t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return {"n": len(values), "mean": avg, "sample_std": sd,
            "ci95_low": avg - half, "ci95_high": avg + half}


def read_runs(paths):
    runs = []
    for path in sorted(set(Path(p).resolve() for p in paths)):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "completed":
            raise ValueError(f"Run is incomplete: {path}")
        if any(not value.get(k) for k in IDENTITY):
            raise ValueError(f"Run lacks dataset/source/selection/objective: {path}")
        if type(value.get("seed")) is not int:
            raise ValueError(f"Run has no integer seed: {path}")
        accuracy = value.get("test_accuracy")
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
            raise ValueError(f"Expected test_accuracy in [0,1]: {path}")
        fps = value.get("fingerprints", {})
        if any(not isinstance(fps.get(k), str) or not fps[k] for k in CONSISTENT):
            raise ValueError(f"Run lacks required input/protocol fingerprints: {path}")
        value["metrics_path"] = str(path)
        value["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        runs.append(value)
    if not runs:
        raise ValueError("No completed controlled metrics found")
    return runs


def summarize(runs, required_seeds=(42, 43, 44)):
    groups = {}
    # A separate correction seed identifies mask variation; it is never counted
    # as an additional independent training seed for the same fixed-mask run.
    for run in runs:
        key = tuple(run[k] for k in IDENTITY) + (run.get("selection_seed"),)
        groups.setdefault(key, []).append(run)
    summaries, indexed = [], {}
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds = [row["seed"] for row in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate training seed for {key}: {seeds}")
        if not set(required_seeds).issubset(seeds):
            raise ValueError(f"Missing required training seeds for {key}: {sorted(set(required_seeds) - set(seeds))}")
        for field in CONSISTENT:
            if len({row["fingerprints"][field] for row in rows}) != 1:
                raise ValueError(f"Mixed {field} within fixed-mask seed group {key}")
        ordered = sorted(rows, key=lambda r: r["seed"])
        summary = {**dict(zip(IDENTITY, key[:4])), "selection_seed": key[4],
                   **mean_interval([row["test_accuracy"] * 100 for row in ordered]),
                   "units": "percentage_points", "seeds": sorted(seeds),
                   "fingerprints": {field: rows[0]["fingerprints"][field] for field in CONSISTENT},
                   "runs": [{k: row[k] for k in ("seed", "test_accuracy", "metrics_path", "metrics_sha256")}
                            for row in ordered]}
        summaries.append(summary)
        indexed[key] = {row["seed"]: row for row in rows}
    comparisons = []
    # Pair the two objectives only when the exact data, mask, protocol,
    # initialization and batch sequence fingerprints agree seed by seed.
    for key, left in indexed.items():
        if key[3] != "fcl_auc":
            continue
        right_key = key[:3] + ("ce_soft", key[4])
        right = indexed.get(right_key)
        if right is None:
            continue
        if set(left) != set(right):
            raise ValueError(f"Paired objectives have different seed sets: {key}")
        for seed in left:
            for field in (*CONSISTENT, "init_state_sha256", "sampler_sha256"):
                lval, rval = left[seed]["fingerprints"].get(field), right[seed]["fingerprints"].get(field)
                if not lval or lval != rval:
                    raise ValueError(f"Objective pairing differs at seed {seed}, {field}: {key}")
        differences = [(left[seed]["test_accuracy"] - right[seed]["test_accuracy"]) * 100
                       for seed in sorted(left)]
        comparisons.append({"dataset": key[0], "label_source": key[1], "selection": key[2],
                            "selection_seed": key[4], "contrast": "fcl_auc - ce_soft",
                            "seeds": sorted(left), "units": "percentage_points",
                            "per_seed_differences": differences, **mean_interval(differences)})
    return {"schema_version": 1, "status": "completed", "required_training_seeds": list(required_seeds),
            "method": "sample SD (ddof=1); unbounded two-sided 95% Student-t CI; paired objective differences",
            "groups": summaries, "paired_objective_differences": comparisons}


def write_summary(result, destination):
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite summary: {destination}")
    (destination / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    columns = [*IDENTITY, "selection_seed", "n", "mean", "sample_std", "ci95_low", "ci95_high", "seeds"]
    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "seeds": ",".join(map(str, row["seeds"]))} for row in result["groups"])
    lines = ["# Controlled training results", "", result["method"] + ". All accuracy values below are percentages.", "",
             "| Dataset | Labels | Selection | Mask seed | Objective | Seeds | Mean ± sample SD | 95% CI |",
             "|---|---|---|---|---|---|---|---|"]
    for row in result["groups"]:
        lines.append(f"| {row['dataset']} | {row['label_source']} | {row['selection']} | {row['selection_seed']} | "
                     f"{row['objective']} | {row['seeds']} | {row['mean']:.4f} ± {row['sample_std']:.4f} | "
                     f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |")
    if result["paired_objective_differences"]:
        lines.extend(["", "## Paired objective differences", "", "Difference in percentage points, fcl_auc − ce_soft.", "",
                      "| Dataset | Labels / selection | Mask seed | Mean difference ± sample SD | 95% CI |",
                      "|---|---|---|---|---|"])
        for row in result["paired_objective_differences"]:
            lines.append(f"| {row['dataset']} | {row['label_source']} / {row['selection']} | {row['selection_seed']} | "
                         f"{row['mean']:.4f} ± {row['sample_std']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |")
    (destination / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if len(set(args.required_seeds)) < 3 or len(set(args.required_seeds)) != len(args.required_seeds):
        parser.error("--required-seeds needs at least three distinct training seeds")
    result = summarize(read_runs(args.runs_root.rglob("metrics.json")), args.required_seeds)
    write_summary(result, args.output_dir)
    print(f"Saved {len(result['groups'])} fixed-mask groups to {args.output_dir}")


if __name__ == "__main__":
    main()
