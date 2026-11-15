"""Audit source labels and save fixed prototype/random oracle-correction masks.

Random ``global_exact`` uses exactly floor(rho * N + .5) rows, independently
of the prototype budget. Prototype correction uses ceil(rho * n) per predicted
class, so its actual budget can be larger; a separate matched-budget random
condition is also saved. Ground truth is used only after a mask is selected.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_paper_experiments import qwen_split_metadata
from src.paths import OUTPUT_ROOT, PROCESSED_ROOT

DATASETS = ("cifar100", "caltech101", "stanfordcars", "eurosat", "food101", "tinyimagenet200", "dtd")
MASK_HASH_FORMAT = "torch_dtype + tuple_shape + contiguous_numpy_bytes"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def json_sha256(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _labels(value, name: str, count: int, classes: int) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) != count
            or not torch.isfinite(value).all()):
        raise ValueError(f"Invalid {name}; expected a finite 1-D tensor of length {count}")
    if value.is_complex() or value.dtype == torch.bool or (value.is_floating_point() and not torch.equal(value, value.round())):
        raise ValueError(f"{name} must contain integer class IDs")
    if (value < 0).any() or (value >= classes).any():
        raise ValueError(f"{name} contains labels outside [0, {classes})")
    return value.long()


def _has_source(cache: dict, source: str, split: str) -> bool:
    if source == "qwen":
        return qwen_split_metadata(cache, split) is not None
    metadata = cache.get("annotation_metadata", {})
    metadata = metadata.get(split, {}) if isinstance(metadata, dict) else {}
    if isinstance(metadata, dict) and metadata.get("replaced_clip_fields"):
        models = metadata.get("source_models", [])
        return bool(models) and all("clip" in str(model).lower() for model in models)
    return True  # Dedicated CLIP extractor fields, with no conflicting overlay.


def validate_cache(cache: dict, source: str, prompt: str = "C") -> tuple[dict, dict, dict]:
    """Validate all features/ground truth and only the selected source's priors."""
    if source not in ("clip", "qwen") or not isinstance(cache, dict):
        raise ValueError("Expected a CLIP/Qwen tensor cache dictionary")
    prior = cache.get("clip_probs_train")
    if not isinstance(prior, torch.Tensor) or prior.ndim != 2 or prior.shape[1] < 1:
        raise ValueError("Training cache requires clip_probs_train")
    classes, feature_dim = prior.shape[1], None
    predictions, metrics, omitted = {}, {}, {}
    for split in ("train", "val", "test"):
        features, truth = cache.get(f"{split}_feats"), cache.get(f"{split}_labels")
        if (not isinstance(features, torch.Tensor) or features.ndim != 2 or not features.numel()
                or features.is_complex() or not torch.isfinite(features).all()):
            raise ValueError(f"Invalid {split}_feats")
        if feature_dim is None:
            feature_dim = features.shape[1]
        if features.shape[1] != feature_dim:
            raise ValueError(f"{split} feature dimension differs from train")
        truth = _labels(truth, f"{split}_labels", len(features), classes)
        if not _has_source(cache, source, split):
            omitted[split] = f"No verified {source} provenance for this split's clip_* fields"
            continue
        if source == "qwen":
            variant = qwen_split_metadata(cache, split).get("prompt_variant")
            if variant in ("A", "B", "C") and variant != prompt:
                raise ValueError(f"Qwen {split} prompt {variant} differs from requested {prompt}")
        probabilities = cache.get(f"clip_probs_{split}")
        predicted = cache.get(f"clip_argmax_{split}")
        if probabilities is not None:
            if (not isinstance(probabilities, torch.Tensor) or probabilities.shape != (len(features), classes)
                    or probabilities.is_complex() or not torch.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not torch.allclose(probabilities.float().sum(1), torch.ones(len(features)), atol=1e-3)):
                raise ValueError(f"Invalid {split} source probabilities")
            inferred = probabilities.argmax(1)
            if predicted is not None and not torch.equal(_labels(predicted, f"clip_argmax_{split}", len(features), classes), inferred):
                raise ValueError(f"{split} source argmax/probability disagreement")
            predicted = inferred
        elif predicted is not None:
            predicted = _labels(predicted, f"clip_argmax_{split}", len(features), classes)
        else:
            omitted[split] = f"No {source} source probabilities or predictions"
            continue
        predictions[split] = predicted
        errors = int(predicted.ne(truth).sum())
        metrics[split] = {"accuracy": 1.0 - errors / len(truth), "samples": len(truth), "errors": errors,
                          "prediction_field": f"clip_probs_{split}" if probabilities is not None else f"clip_argmax_{split}"}
    if "train" not in predictions:
        raise ValueError(f"Training requires verified {source} source probabilities; CLIP/Qwen fallback is forbidden")
    return predictions, metrics, omitted


def stable_sample_order(cache: dict, count: int, cache_sha256: str) -> tuple[list[str], str]:
    candidates = []
    for key in ("train_sample_ids", "sample_ids_train"):
        if key in cache:
            candidates.append((key, cache[key]))
    if isinstance(cache.get("sample_ids"), dict) and "train" in cache["sample_ids"]:
        candidates.append(("sample_ids.train", cache["sample_ids"]["train"]))
    kind = "sample_id"
    if not candidates:
        kind = "path"
        for key in ("train_paths", "paths_train"):
            if key in cache:
                candidates.append((key, cache[key]))
        if isinstance(cache.get("sample_paths"), dict) and cache["sample_paths"].get("train"):
            candidates.append(("sample_paths.train", cache["sample_paths"]["train"]))
    values = []
    for name, candidate in candidates:
        if isinstance(candidate, torch.Tensor):
            candidate = candidate.tolist()
        if not isinstance(candidate, (list, tuple)) or len(candidate) != count:
            raise ValueError(f"{name} does not match the training sample count")
        current = [str(value) for value in candidate]
        if any(not value for value in current) or len(set(current)) != count:
            raise ValueError(f"{name} contains empty or duplicate sample identities")
        if values and current != values:
            raise ValueError("Training identity aliases disagree; sample order is ambiguous")
        values = current
    if not values:
        # This is a stable order only within the exact hashed legacy cache.
        values = [f"cache-sha256:{cache_sha256}/train-row:{i}" for i in range(count)]
        kind = "cache_row_index_only"
    return values, kind


def prototype_mask(features: torch.Tensor, pseudo: torch.Tensor, rho: float) -> tuple[torch.Tensor, dict]:
    """Selection deliberately accepts no ground-truth labels."""
    normalized = F.normalize(features.float(), dim=1)
    mask = torch.zeros(len(pseudo), dtype=torch.bool)
    scores = torch.zeros(len(pseudo), dtype=torch.float32)
    class_budgets = {}
    for label in torch.unique(pseudo, sorted=True).tolist():
        indices = torch.where(pseudo.eq(label))[0]
        center = F.normalize(normalized[indices].mean(0), dim=0)
        distances = 1.0 - normalized[indices] @ center
        scores[indices] = distances
        count = int((Decimal(str(rho)) * len(indices)).to_integral_value(rounding=ROUND_CEILING))
        # Equal distances are broken by the original cache row, not RNG state.
        selected = indices[torch.argsort(distances, descending=True, stable=True)[:count]]
        mask[selected] = True
        class_budgets[str(label)] = {"samples": len(indices), "selected": count}
    return mask, {"outlier_scores": scores, "class_budgets": class_budgets}


def random_mask(count: int, budget: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(count, dtype=torch.bool)
    mask[torch.randperm(count, generator=generator)[:budget]] = True
    return mask


def correction_statistics(pseudo: torch.Tensor, truth: torch.Tensor, selected: torch.Tensor) -> dict:
    wrong = pseudo.ne(truth)
    selected_count, wrong_count = int(selected.sum()), int(wrong.sum())
    detected = int((selected & wrong).sum())
    remaining = len(truth) - selected_count
    return {"source_accuracy": 1.0 - wrong_count / len(truth),
            "corrected_label_accuracy": 1.0 - (wrong_count - detected) / len(truth),
            "corrected_count": selected_count, "corrected_ratio": selected_count / len(truth),
            "source_errors": wrong_count, "selected_errors": detected,
            "selected_error_precision": detected / selected_count if selected_count else None,
            "selected_error_recall": detected / wrong_count if wrong_count else None,
            "uncorrected_error_rate": (wrong_count - detected) / remaining if remaining else None}


def compare_cache(cache_path: Path, source: str, dataset: str, output_dir: Path,
                  rho: float = .3, seeds=(42, 43, 44), prompt: str = "C") -> dict:
    if not math.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must be finite and in (0, 1]")
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Selection seeds must be unique nonnegative integers")
    cache_path, output_dir = Path(cache_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Use a new output directory to preserve fixed masks: {output_dir}")
    cache_sha = file_sha256(cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    predictions, source_metrics, omitted = validate_cache(cache, source, prompt)
    pseudo, truth, features = predictions["train"], cache["train_labels"].long(), cache["train_feats"]
    sample_ids, order_kind = stable_sample_order(cache, len(pseudo), cache_sha)
    sample_order_sha = json_sha256(sample_ids)
    prototype, prototype_details = prototype_mask(features, pseudo, rho)
    prototype_budget = int(prototype.sum())
    exact_budget = int((Decimal(str(rho)) * len(pseudo) + Decimal(".5")).to_integral_value(rounding=ROUND_FLOOR))
    conditions = [("prototype", None, prototype, "sum_over_predicted_classes(ceil(rho*n_class))")]
    for seed in seeds:
        conditions.extend([
            ("random_global_exact", seed, random_mask(len(pseudo), exact_budget, seed), "floor(rho*N+.5)"),
            ("random_matched_prototype", seed, random_mask(len(pseudo), prototype_budget, seed), "prototype_actual_count"),
        ])
    output_dir.mkdir(parents=True)
    (output_dir / "sample_order.json").write_text(json.dumps({"kind": order_kind, "sample_ids": sample_ids,
        "sample_order_sha256": sample_order_sha, "cache_sha256": cache_sha}, ensure_ascii=False, indent=2), encoding="utf-8")
    shared_meta = {"schema_version": 1, "dataset": dataset, "label_source": source,
        "prompt": prompt if source == "qwen" else None, "from_cache": str(cache_path.resolve()),
        "cache_sha256": cache_sha, "sample_order_sha256": sample_order_sha, "sample_order_kind": order_kind,
        "N": len(pseudo), "top_outlier_pct": rho, "requested_rho": rho,
        "source_probability_field": "clip_probs_train", "mask_sha256_format": MASK_HASH_FORMAT,
        "correction": "oracle simulation using dataset ground truth only at selected indices",
        "selection_uses_ground_truth": False}
    rows = []
    for method, seed, mask, budget_rule in conditions:
        filename = f"{method}.pt" if seed is None else f"{method}_seed-{seed}.pt"
        observed = pseudo.clone()
        observed[mask] = truth[mask]
        flags = (~mask).long()
        statistics_row = correction_statistics(pseudo, truth, mask)
        metadata = {**shared_meta, "method": method, "selection": method, "selection_seed": seed, "budget_rule": budget_rule,
                    "mask_sha256": tensor_sha256(mask), "y_obs_sha256": tensor_sha256(observed), **statistics_row}
        payload = {"y_obs": observed, "s": flags, "y_pseudo": pseudo.clone(), "selected_mask": mask,
                   "train_sample_ids": sample_ids, "train_row_indices": torch.arange(len(pseudo)), "meta": metadata}
        if method == "prototype":
            payload.update(outlier_scores=prototype_details["outlier_scores"])
            metadata["predicted_class_budgets"] = prototype_details["class_budgets"]
        torch.save(payload, output_dir / filename)
        rows.append({"method": method, "selection_seed": seed, "observed_labels": str((output_dir / filename).resolve()),
                     "budget_rule": budget_rule, "mask_sha256": metadata["mask_sha256"], **statistics_row})
    aggregates = {}
    for method in ("random_global_exact", "random_matched_prototype"):
        members = [row for row in rows if row["method"] == method]
        aggregates[method] = {}
        for metric in ("corrected_label_accuracy", "selected_error_precision", "selected_error_recall"):
            values = [row[metric] for row in members if row[metric] is not None]
            aggregates[method][metric] = {"mean": statistics.mean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None, "seeds": len(values)}
    report = {**shared_meta, "script_sha256": file_sha256(Path(__file__)),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "source_metrics": source_metrics,
        "omitted_source_splits": omitted, "random_selection_seeds": seeds,
        "prototype_actual_budget": prototype_budget, "random_global_exact_budget": exact_budget,
        "global_requested_ratio_is_exactly_representable": Decimal(str(rho)) * len(pseudo) == exact_budget,
        "budget_explanation": "Random global_exact fixes floor(rho*N+.5) rows; when rho*N is not an integer the requested fraction is rounded to the nearest row. Prototype uses ceil per predicted class. random_matched_prototype matches its actual count and is a separate comparison.",
        "fixed_prototype_observed": str((output_dir / "prototype.pt").resolve()),
        "controlled_use": {"ce_vs_soft_ce": "Reuse prototype.pt unchanged for both objectives",
                           "random_auc_c": "Use random_global_exact_seed-<seed>.pt; selection_seed and training seed are separate settings"},
        "conditions": rows, "random_aggregates": aggregates}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--source", choices=("clip", "qwen"), default="clip")
    parser.add_argument("--prompt", choices=("A", "B", "C"), default="C")
    parser.add_argument("--cache", type=Path, help="Explicit cache for a single selected dataset")
    parser.add_argument("--cache-template", help="Fields: processed,dataset,source,prompt")
    parser.add_argument("--rho", type=float, default=.3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Mask selection seeds")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "label_corrections")
    parser.add_argument("--output-dir", type=Path, help="Explicit new directory for a single selected dataset")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if (args.cache or args.output_dir) and len(args.datasets) != 1:
        parser.error("--cache/--output-dir require exactly one --datasets value")
    default_template = ("{processed}/{dataset}_clip_vit-l-14_embeddings.pt" if args.source == "clip"
                        else "{processed}/{dataset}_qwen_prompt-{prompt}_embeddings.pt")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    failures = 0
    for dataset in args.datasets:
        cache = args.cache or Path((args.cache_template or default_template).format(
            processed=PROCESSED_ROOT, dataset=dataset, source=args.source, prompt=args.prompt))
        output = args.output_dir or args.output_root / dataset / args.source / timestamp
        try:
            report = compare_cache(cache, args.source, dataset, output, args.rho, args.seeds, args.prompt)
            print(json.dumps({"dataset": dataset, "source": args.source, "report": str(output / "report.json"),
                "source_metrics": report["source_metrics"], "prototype_budget": report["prototype_actual_budget"],
                "random_exact_budget": report["random_global_exact_budget"]}, ensure_ascii=False), flush=True)
        except (OSError, ValueError, KeyError) as exc:
            failures += 1
            print(f"[ERROR] {dataset}/{args.source}: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                return 1
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
