"""Sequential, resumable Qwen labeling; raw outputs and split provenance are retained."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from paths import PROCESSED_ROOT, OUTPUT_ROOT
from utils.annotation.schema import AnnotationError, Manifest, fingerprint, make_record, parse_label, read_classes
from utils.annotation.store import AnnotationStore
from utils.annotation.cli import export_cache
from scripts.prepare_annotation_manifests import sha256_file


DATASET_NAMES = {"cifar100": "CIFAR-100", "caltech101": "Caltech-101", "eurosat": "EuroSAT",
                 "food101": "Food-101", "tinyimagenet200": "Tiny-ImageNet-200",
                 "stanfordcars": "Stanford Cars", "dtd": "Describable Textures Dataset (DTD)"}


def appendix_prompt(dataset, classes):
    menu = "\n".join(f"{index}: {name.replace('_', ' ')}" for index, name in enumerate(classes))
    return (f"You are a classifier for the {DATASET_NAMES.get(dataset, dataset)}. Given the image, "
            f"choose exactly ONE class from the list below.\n{menu}\nReply with ONLY the integer id.")


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def labeling_source_fingerprint():
    files = [Path(__file__), ROOT / "scripts/prepare_annotation_manifests.py"]
    files.extend(sorted((ROOT / "src/utils/annotation").glob("*.py")))
    return fingerprint({str(path.relative_to(ROOT)): sha256_file(path) for path in files})


def run(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config["labeling"]
    if settings.get("decoding") != "greedy_legal_class_id_token_trie":
        raise ValueError("The controlled campaign requires explicit legal-class-ID constrained decoding")
    datasets = args.datasets or config["datasets"]
    splits = args.splits or settings["splits"]
    report = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "config": config,
              "selected_datasets": datasets, "selected_splits": splits,
              "model": settings["model"], "datasets": {}, "status": "running"}
    predictor = None
    environment_path = args.output_root / "model_environment.json"
    pinned_environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else None
    revision = (pinned_environment or {}).get("resolved_commit") or settings["revision"]
    source_hash = labeling_source_fingerprint()
    if pinned_environment and pinned_environment.get("labeling_source_sha256") != source_hash:
        raise ValueError("Labeling source changed; use a new campaign directory")

    def ensure_predictor():
        nonlocal predictor, pinned_environment
        if predictor is not None:
            return
        from utils.annotation.qwen_batch import BatchedQwenPredictor
        predictor = BatchedQwenPredictor(args.model_path or settings["model"], revision,
            settings["min_pixels"], settings["max_pixels"], settings["dtype"], settings["attention"])
        predictor.metadata.update(requested_revision=settings["revision"], configured_model=settings["model"],
                                  labeling_source_sha256=source_hash)
        if args.model_path:
            weights = {str(path.relative_to(args.model_path)): sha256_file(path) for path in
                       sorted(args.model_path.rglob("*")) if path.is_file()
                       and not any(part.startswith(".") for part in path.relative_to(args.model_path).parts)}
            if not weights:
                raise ValueError("Local model directory contains no model/config/tokenizer files")
            predictor.metadata["local_files_sha256"] = weights
        stable_environment = json.loads(json.dumps(predictor.metadata, default=str))
        if pinned_environment and stable_environment != pinned_environment:
            raise ValueError("Model weights/config or inference environment changed during resume")
        pinned_environment = stable_environment
        dump(environment_path, stable_environment)

    for dataset in datasets:
        manifest_dir = args.manifest_root / dataset
        manifest_path, classes_path = manifest_dir / "manifest.jsonl", manifest_dir / "classes.json"
        cache_path = args.processed_root / f"{dataset}_clip_vit-l-14_embeddings.pt"
        for required in (manifest_path, classes_path, manifest_dir / "manifest_metadata.json", cache_path):
            if not required.is_file():
                raise FileNotFoundError(f"Dataset preparation is incomplete: {required}")
        manifest_meta = json.loads((manifest_dir / "manifest_metadata.json").read_text(encoding="utf-8"))
        if manifest_meta["cache_sha256"] != sha256_file(cache_path):
            raise ValueError(f"Feature cache changed after manifest creation: {dataset}")
        classes = read_classes(classes_path)
        if (Manifest.read(manifest_path).sha256 != manifest_meta["manifest_sha256"]
                or fingerprint(classes) != manifest_meta["class_names_sha256"]):
            raise ValueError(f"Manifest or ordered class vocabulary changed: {dataset}")
        prompt = appendix_prompt(dataset, classes)
        result_dir = args.output_root / dataset
        result_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = result_dir / "prompt.txt"
        if prompt_path.exists() and prompt_path.read_text(encoding="utf-8") != prompt:
            raise ValueError("Existing prompt changed; choose a new campaign output directory")
        prompt_path.write_text(prompt, encoding="utf-8")
        dataset_report = {"status": "running", "splits": {}, "prompt_sha256": fingerprint(prompt),
                          "cache_sha256": manifest_meta["cache_sha256"], "source_model": settings["model"],
                          "decoding": settings["decoding"]}
        report["datasets"][dataset] = dataset_report
        cache_for_overlay = cache_path
        import torch
        torch.manual_seed(settings["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings["seed"])
        truth_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        for split in splits:
            manifest = Manifest.read(manifest_path, split)
            progress = result_dir / f"{split}.jsonl"
            if pinned_environment is None:
                ensure_predictor()
            annotation_settings = {"mode": "qwen", "backend": "qwen2.5-vl",
                "prompt_variant": "C", "prompt_sha256": fingerprint(prompt), "seed": settings["seed"],
                "requested_revision": settings["revision"], "min_pixels": settings["min_pixels"],
                "max_pixels": settings["max_pixels"], "dtype": settings["dtype"],
                "attention": settings["attention"], "max_new_tokens": settings["max_new_tokens"],
                "do_sample": False, "batch_size": settings["batch_size"],
                "decoding": settings["decoding"],
                "allowed_class_count": len(classes),
                "effective_generation": {"num_beams": 1, "num_return_sequences": 1, "min_length": 0,
                    "min_new_tokens": 0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "use_cache": True},
                "cache_sha256": manifest_meta["cache_sha256"],
                "model_environment_sha256": fingerprint(pinned_environment)}
            store = AnnotationStore(progress, manifest, classes, settings["model"], annotation_settings,
                                    resume=args.resume)
            todo = [row for row in manifest.samples if row["sample_id"] not in store.records]
            if args.limit is not None:
                todo = todo[:args.limit]
            if todo and predictor is None:
                ensure_predictor()
            started = time.monotonic()
            for start in range(0, len(todo), settings["batch_size"]):
                batch = todo[start:start + settings["batch_size"]]
                raw_outputs = predictor([row["path"] for row in batch], prompt, settings["max_new_tokens"], classes=classes)
                if len(raw_outputs) != len(batch):
                    raise ValueError("Qwen returned an unexpected batch length")
                rejected = []
                for sample, raw in zip(batch, raw_outputs):
                    try:
                        label = parse_label(raw.strip(), classes)
                        store.append(make_record(sample, label, classes, settings["model"], raw_output=raw))
                    except AnnotationError as error:
                        rejected.append({**sample, "raw_output": raw, "error": str(error)})
                if rejected:
                    with (result_dir / f"{split}.errors.jsonl").open("a", encoding="utf-8") as handle:
                        for rejected_row in rejected:
                            handle.write(json.dumps(rejected_row, ensure_ascii=False) + "\n")
                    raise AnnotationError(f"Qwen returned {len(rejected)} invalid labels; successful labels saved, "
                                          f"no guess/fallback applied: {result_dir / (split + '.errors.jsonl')}")
                dataset_report["splits"][split] = {"annotated": len(store.records), "expected": len(manifest.samples),
                    "elapsed_this_invocation_seconds": time.monotonic() - started, "status": "running"}
                dump(args.output_root / "status.json", report)
                print(f"[QWEN] {dataset}/{split} {len(store.records)}/{len(manifest.samples)} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
            if len(store.records) != len(manifest.samples):
                dataset_report["splits"][split] = {"annotated": len(store.records), "expected": len(manifest.samples),
                                                    "status": "partial"}
                report["status"] = "partial"
                dump(args.output_root / "status.json", report)
                return
            predictions = torch.tensor([store.records[row["sample_id"]]["label_id"] for row in manifest.samples])
            truth = truth_cache[f"{split}_labels"].long()
            accuracy = float(predictions.eq(truth).float().mean())
            dataset_report["splits"][split] = {"annotated": len(predictions), "expected": len(truth),
                "correct": int(predictions.eq(truth).sum()), "accuracy": accuracy,
                "annotations_sha256": sha256_file(progress), "status": "completed"}
            overlay_path = result_dir / f"{dataset}_qwen_through-{split}.pt"
            overlay_provenance_path = overlay_path.with_suffix(".provenance.json")
            overlay_inputs = {"original_cache_sha256": manifest_meta["cache_sha256"],
                "parent_cache_sha256": sha256_file(cache_for_overlay),
                "annotations_sha256": sha256_file(progress), "manifest_sha256": manifest.sha256,
                "prompt_sha256": fingerprint(prompt), "source_model": settings["model"]}
            if not overlay_path.exists():
                export_cache(argparse.Namespace(manifest=str(manifest_path), classes=str(classes_path),
                    annotations=str(progress), cache=str(cache_for_overlay), output=str(overlay_path), split=split,
                    confirm_cache_manifest_order=bool(manifest_meta.get("legacy_cache_order_verified")),
                    replace_clip_fields=True))
                dump(overlay_provenance_path, {"inputs": overlay_inputs, "output_sha256": sha256_file(overlay_path)})
            else:
                if not overlay_provenance_path.exists():
                    raise ValueError(f"Existing Qwen overlay has no provenance: {overlay_path}")
                prior_provenance = json.loads(overlay_provenance_path.read_text(encoding="utf-8"))
                if prior_provenance.get("inputs") != overlay_inputs or prior_provenance.get("output_sha256") != sha256_file(overlay_path):
                    raise ValueError(f"Existing Qwen overlay/input fingerprint mismatch: {overlay_path}")
                previous = torch.load(overlay_path, map_location="cpu", weights_only=True)
                meta = previous.get("annotation_metadata", {}).get(split, {})
                if not torch.equal(previous[f"clip_argmax_{split}"], predictions) or meta.get("source_models") != [settings["model"]]:
                    raise ValueError(f"Existing Qwen overlay differs from annotations: {overlay_path}")
            # Keep one cumulative cache so all labelled split fields remain Qwen.
            cache_for_overlay = overlay_path
            dump(result_dir / "label_accuracy.json", dataset_report)
            dump(args.output_root / "status.json", report)
        dataset_report.update(status="completed", derived_cache=str(cache_for_overlay),
                              derived_cache_sha256=sha256_file(cache_for_overlay))
        dump(result_dir / "label_accuracy.json", dataset_report)
        dump(args.output_root / "status.json", report)
    report.update(status="completed", ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
    dump(args.output_root / "status.json", report)
    prior_failure = args.output_root / "failure.json"
    if prior_failure.exists():
        prior = json.loads(prior_failure.read_text(encoding="utf-8"))
        prior.update(resolved=True, resolved_at=report["ended_at"])
        dump(prior_failure, prior)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/controlled_campaign.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"))
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=PROCESSED_ROOT / "annotation_manifests")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "controlled_correction_20261115" / "qwen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-path", type=Path, help="Verified local snapshot of the configured official model.")
    parser.add_argument("--limit", type=int, help="Smoke only; incomplete labels are never exported or reported as full results.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        run(args)
    except Exception as error:
        dump(args.output_root / "failure.json", {"status": "failed", "error": repr(error),
            "time": dt.datetime.now(dt.timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
