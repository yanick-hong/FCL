"""Plan or execute ordered, fingerprint-checked correction/training stages.

Dry-run is the default. Qwen inference is a prerequisite, performed separately
by run_qwen_campaign.py. Completed work is reused only after checking inputs,
source, protocol and the immutable output inventory; partial work is preserved
and causes a clear failure instead of being overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from paths import OUTPUT_ROOT, PROCESSED_ROOT
from scripts import compare_label_corrections as correction
from scripts import run_controlled_training as training
from scripts import run_qwen_campaign as qwen
from scripts import summarize_controlled_results as statistics
from utils.annotation.schema import Manifest, fingerprint, read_classes
from utils.annotation.store import load_annotations

PHASES = ("corrections", "train_random_auc", "train_soft_ce", "train_core", "summary")
RECEIPT = ".suite_receipt.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/controlled_campaign.yaml")
    parser.add_argument("--paper-config", type=Path, default=ROOT / "configs/paper_reproduction.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--qwen-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="Default; print the job list without creating outputs")
    return parser.parse_args(argv)


def configure(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paper = yaml.safe_load(args.paper_config.read_text(encoding="utf-8"))
    datasets = args.datasets or config["datasets"]
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets).issubset(config["datasets"]):
        raise ValueError("Datasets must be distinct members of the campaign config")
    if not args.phases or len(set(args.phases)) != len(args.phases):
        raise ValueError("Phases must be distinct")
    seeds = config["training"]["seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("At least three distinct nonnegative training seeds are required")
    mask_seeds = config["correction"]["random_mask_seeds"]
    if 42 not in mask_seeds or len(set(mask_seeds)) != len(mask_seeds):
        raise ValueError("Correction mask seeds must be distinct and include the fixed random mask seed 42")
    if config["training"].get("patience", 0) != 0:
        raise ValueError("The controlled suite requires fixed-epoch training (patience=0)")
    args.output_root = args.output_root or OUTPUT_ROOT / config["campaign"]
    args.qwen_root = args.qwen_root or args.output_root / "qwen"
    args.manifest_root = args.manifest_root or args.processed_root / "annotation_manifests"
    return config, paper, datasets


def cache_path(args, dataset, source):
    return (args.processed_root / f"{dataset}_clip_vit-l-14_embeddings.pt" if source == "clip"
            else args.qwen_root / dataset / f"{dataset}_qwen_through-test.pt")


def corrections_dir(args, dataset, source):
    return args.output_root / "corrections" / dataset / source


def build_jobs(args, config, paper, datasets):
    jobs = []
    for phase in PHASES:
        if phase not in args.phases:
            continue
        if phase == "summary":
            jobs.append({"id": "summary", "phase": phase, "kind": "summary",
                         "output": str(args.output_root / "summaries" / ("__".join(sorted(datasets))))})
            continue
        for dataset in datasets:
            conditions = ([("clip", None, None), ("qwen", None, None)] if phase == "corrections" else
                          [("clip", "random_global_exact", "fcl_auc")] if phase == "train_random_auc" else
                          [("clip", "prototype", "ce_soft")] if phase == "train_soft_ce" else
                          [("clip", "prototype", "fcl_auc"), ("qwen", "prototype", "fcl_auc")])
            for source, selection, objective in conditions:
                cache = cache_path(args, dataset, source)
                if phase == "corrections":
                    output = corrections_dir(args, dataset, source)
                    command = [sys.executable, str(ROOT / "scripts/compare_label_corrections.py"),
                               "--datasets", dataset, "--source", source, "--cache", str(cache),
                               "--output-dir", str(output), "--rho", str(config["correction"]["rho"]),
                               "--seeds", *map(str, config["correction"]["random_mask_seeds"]), "--prompt", "C"]
                    jobs.append({"id": f"corrections/{dataset}/{source}", "phase": phase, "kind": "correction",
                                 "dataset": dataset, "source": source, "cache": str(cache),
                                 "output": str(output), "command": command})
                    continue
                pk = paper["datasets"].get(dataset, {"P": 16, "K": 16} if dataset == "dtd" else {})
                if not pk.get("P") or not pk.get("K"):
                    raise ValueError(f"Missing recovered P/K settings: {dataset}")
                filename = "random_global_exact_seed-42.pt" if selection == "random_global_exact" else "prototype.pt"
                obs = corrections_dir(args, dataset, source) / filename
                for seed in config["training"]["seeds"]:
                    output = args.output_root / "runs" / dataset / source / selection / objective / f"seed-{seed}"
                    command = [sys.executable, str(ROOT / "scripts/run_controlled_training.py"),
                               "--cache", str(cache), "--obs-labels", str(obs), "--output-dir", str(output),
                               "--dataset", dataset, "--objective", objective, "--seed", str(seed),
                               "--P", str(pk["P"]), "--K", str(pk["K"]), "--device", args.device]
                    for key in ("epochs", "lr", "lr_max", "weight_decay", "lam", "mu", "tau", "num_workers"):
                        command.extend(["--" + key.replace("_", "-"), str(config["training"][key])])
                    jobs.append({"id": f"{phase}/{dataset}/{source}/seed-{seed}", "phase": phase, "kind": "training",
                                 "dataset": dataset, "source": source, "selection": selection,
                                 "selection_seed": 42 if selection == "random_global_exact" else None,
                                 "objective": objective, "seed": seed, "cache": str(cache), "obs": str(obs),
                                 "output": str(output), "command": command})
    return jobs


def verify_qwen(args, config, dataset):
    """Require all three complete inference artifacts and their provenance chain."""
    base = cache_path(args, dataset, "clip")
    base_hash = correction.file_sha256(base)
    directory = args.qwen_root / dataset
    environment = read_json(args.qwen_root / "model_environment.json")
    if (environment.get("configured_model") != config["labeling"]["model"]
            or environment.get("labeling_source_sha256") != qwen.labeling_source_fingerprint()):
        raise ValueError(f"Qwen model/source fingerprint mismatch: {dataset}")
    manifest_dir = args.manifest_root / dataset
    manifest_path, classes_path = manifest_dir / "manifest.jsonl", manifest_dir / "classes.json"
    manifest_meta, classes = read_json(manifest_dir / "manifest_metadata.json"), read_classes(classes_path)
    if (manifest_meta["cache_sha256"] != base_hash or Manifest.read(manifest_path).sha256 != manifest_meta["manifest_sha256"]
            or fingerprint(classes) != manifest_meta["class_names_sha256"]):
        raise ValueError(f"Qwen manifest/cache/vocabulary fingerprint mismatch: {dataset}")
    prompt_hash = fingerprint(qwen.appendix_prompt(dataset, classes))
    report = read_json(directory / "label_accuracy.json")
    if (report.get("status") != "completed" or report.get("cache_sha256") != base_hash
            or report.get("prompt_sha256") != prompt_hash or report.get("source_model") != config["labeling"]["model"]):
        raise ValueError(f"Qwen label report is incomplete or mismatched: {dataset}")
    parent_hash = base_hash
    for split in ("train", "val", "test"):
        manifest = Manifest.read(manifest_path, split)
        annotations = directory / f"{split}.jsonl"
        annotation_meta = read_json(annotations.with_suffix(".jsonl.meta.json"))
        settings = annotation_meta.get("settings", {})
        expected_settings = {key: config["labeling"][key] for key in
                             ("seed", "min_pixels", "max_pixels", "dtype", "attention", "max_new_tokens", "batch_size")}
        expected_settings.update(requested_revision=config["labeling"]["revision"], do_sample=False,
                                 mode="qwen", backend="qwen2.5-vl", prompt_variant="C",
                                 decoding=config["labeling"]["decoding"])
        if (settings.get("model_environment_sha256") != fingerprint(environment)
                or settings.get("cache_sha256") != base_hash or settings.get("prompt_sha256") != prompt_hash
                or any(settings.get(key) != value for key, value in expected_settings.items())):
            raise ValueError(f"Qwen annotation model/input identity mismatch: {dataset}/{split}")
        records = load_annotations(annotations, manifest, classes, config["labeling"]["model"])
        split_report = report.get("splits", {}).get(split, {})
        annotations_hash = correction.file_sha256(annotations)
        if (len(records) != len(manifest.samples) or split_report.get("status") != "completed"
                or split_report.get("annotated") != len(records) or split_report.get("expected") != len(records)
                or split_report.get("annotations_sha256") != annotations_hash):
            raise ValueError(f"Qwen requires complete verified train/val/test labels: {dataset}/{split}")
        overlay = directory / f"{dataset}_qwen_through-{split}.pt"
        provenance = read_json(overlay.with_suffix(".provenance.json"))
        expected = {"original_cache_sha256": base_hash, "parent_cache_sha256": parent_hash,
                    "annotations_sha256": annotations_hash, "manifest_sha256": manifest.sha256,
                    "prompt_sha256": prompt_hash, "source_model": config["labeling"]["model"]}
        if any(provenance.get("inputs", {}).get(key) != value for key, value in expected.items()):
            raise ValueError(f"Qwen overlay input fingerprint mismatch: {dataset}/{split}")
        parent_hash = correction.file_sha256(overlay)
        if provenance.get("output_sha256") != parent_hash:
            raise ValueError(f"Qwen overlay output fingerprint mismatch: {dataset}/{split}")
    if report.get("derived_cache_sha256") != parent_hash:
        raise ValueError(f"Qwen final cache report fingerprint mismatch: {dataset}")
    return directory / f"{dataset}_qwen_through-test.pt"


def output_inventory(directory):
    return {str(path.relative_to(directory)): correction.file_sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != RECEIPT}


def receipt_matches(directory, expected):
    receipt = directory / RECEIPT
    if not receipt.is_file():
        raise FileExistsError(f"Existing output lacks a completed suite receipt; preserved: {directory}")
    stored = read_json(receipt)
    if stored.get("status") != "completed" or stored.get("inputs") != expected:
        raise ValueError(f"Existing job input/source/protocol fingerprints differ: {directory}")
    if stored.get("outputs") != output_inventory(directory):
        raise ValueError(f"Existing job output fingerprint mismatch: {directory}")


def seal_receipt(directory, inputs):
    training.write_json(directory / RECEIPT,
                        {"status": "completed", "inputs": inputs, "outputs": output_inventory(directory)})


def execute_command(command):
    subprocess.run(command, cwd=ROOT, check=True)


def correction_job(job, config):
    directory, cache = Path(job["output"]), Path(job["cache"])
    protocol = {"rho": config["correction"]["rho"], "seeds": config["correction"]["random_mask_seeds"],
                "prototype_budget": "per-class ceil", "random_budget": "floor(rho*N+.5)", "prompt": "C"}
    expected = {"dataset": job["dataset"], "label_source": job["source"],
                "cache_sha256": correction.file_sha256(cache), "protocol_sha256": fingerprint(protocol),
                "source_sha256": fingerprint({str(path.relative_to(ROOT)): correction.file_sha256(path)
                    for path in (Path(correction.__file__), ROOT / "scripts/run_paper_experiments.py")})}
    if directory.exists():
        receipt_matches(directory, expected)
        return True
    execute_command(job["command"])
    report = read_json(directory / "report.json")
    if (report.get("cache_sha256") != expected["cache_sha256"] or report.get("dataset") != job["dataset"]
            or report.get("label_source") != job["source"] or report.get("script_sha256") != correction.file_sha256(Path(correction.__file__))
            or report.get("random_selection_seeds") != protocol["seeds"] or report.get("requested_rho") != protocol["rho"]):
        raise ValueError(f"New correction result does not match the requested job: {directory}")
    for row in report["conditions"]:
        artifact = directory / Path(row["observed_labels"]).name
        checker = training.parse_args(["--cache", str(cache), "--obs-labels", str(artifact),
                                       "--output-dir", str(directory / "unused"), "--objective", "fcl_auc"])
        _, observed, hashes = training.load_fixed_inputs(checker)
        if observed["meta"]["method"] != row["method"] or hashes["mask_sha256"] != row["mask_sha256"]:
            raise ValueError(f"Correction output metadata/mask mismatch: {artifact}")
    seal_receipt(directory, expected)
    return False


def training_job(job):
    args = training.parse_args(job["command"][2:])
    _, observed, hashes = training.load_fixed_inputs(args)
    meta = observed["meta"]
    if (meta["dataset"] != job["dataset"] or meta["label_source"] != job["source"]
            or meta["method"] != job["selection"] or meta.get("selection_seed") != job["selection_seed"]):
        raise ValueError(f"Training requires the planned fixed correction mask: {job['id']}")
    hashes.update(source_sha256=training.source_fingerprint(),
                  protocol_sha256=fingerprint(training.common_protocol(args)))
    identity = {"dataset": job["dataset"], "label_source": job["source"], "selection": job["selection"],
                "selection_seed": job["selection_seed"], "objective": job["objective"], "seed": job["seed"]}
    directory = Path(job["output"])
    reused = directory.exists()
    if reused and not (directory / "metrics.json").is_file():
        raise FileExistsError(f"Incomplete training output preserved; use a new campaign directory: {directory}")
    if not reused:
        execute_command(job["command"])
    result = read_json(directory / "metrics.json")
    if (result.get("status") != "completed" or result.get("completed_epochs") != args.epochs
            or any(result.get(key) != value for key, value in identity.items())
            or any(result.get("fingerprints", {}).get(key) != value for key, value in hashes.items())):
        raise ValueError(f"Training result input/source/protocol fingerprints differ: {directory}")
    for key, filename in (("config_sha256", "config.json"), ("best_ckpt_sha256", "best.ckpt"), ("last_ckpt_sha256", "last.ckpt")):
        if result["fingerprints"].get(key) != correction.file_sha256(directory / filename):
            raise ValueError(f"Training {filename} fingerprint mismatch: {directory}")
    if read_json(directory / "status.json").get("status") != "completed":
        raise ValueError(f"Training status is incomplete: {directory}")
    expected = {"identity": identity, "fingerprints": hashes}
    if (directory / RECEIPT).exists():
        receipt_matches(directory, expected)
    else:
        seal_receipt(directory, expected)
    return reused


def summary_job(job, args, config, paper, datasets):
    all_args = argparse.Namespace(**vars(args))
    all_args.phases = list(PHASES[1:-1])
    jobs = build_jobs(all_args, config, paper, datasets)
    for planned in jobs:
        directory = Path(planned["output"])
        if not directory.is_dir():
            raise FileNotFoundError(f"Summary requires every planned seed: {directory}")
        training_job(planned)
    paths = [Path(planned["output"]) / "metrics.json" for planned in jobs]
    expected = {"metrics": {str(path.resolve()): correction.file_sha256(path) for path in paths},
                "required_seeds": config["training"]["seeds"],
                "source_sha256": correction.file_sha256(Path(statistics.__file__))}
    directory = Path(job["output"])
    if directory.exists():
        receipt_matches(directory, expected)
        return True
    result = statistics.summarize(statistics.read_runs(paths), config["training"]["seeds"])
    statistics.write_summary(result, directory)
    seal_receipt(directory, expected)
    return False


def run(args):
    config, paper, datasets = configure(args)
    jobs = build_jobs(args, config, paper, datasets)
    plan = {"schema_version": 1, "datasets": datasets, "phases": [phase for phase in PHASES if phase in args.phases],
            "config_sha256": correction.file_sha256(args.config), "paper_config_sha256": correction.file_sha256(args.paper_config),
            "suite_source_sha256": correction.file_sha256(Path(__file__)), "jobs": jobs}
    if not args.execute:
        print(json.dumps({"mode": "dry_run", **plan}, ensure_ascii=False, indent=2))
        return plan
    ledger = args.output_root / "orchestration" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ledger.mkdir(parents=True)
    training.write_json(ledger / "plan.json", plan)
    state = {"status": "running", "stages": {phase: "pending" for phase in plan["phases"]}, "jobs": []}
    training.write_json(ledger / "status.json", state)
    verified_qwen = set()
    try:
        for phase in plan["phases"]:
            state["stages"][phase] = "running"
            training.write_json(ledger / "status.json", state)
            if phase in ("corrections", "train_core", "summary"):
                for dataset in datasets:
                    verify_qwen(args, config, dataset)
                    verified_qwen.add(dataset)
            for job in [planned for planned in jobs if planned["phase"] == phase]:
                record = {"id": job["id"], "status": "running", "output": job["output"]}
                state["jobs"].append(record)
                training.write_json(ledger / "status.json", state)
                print(f"[JOB] {job['id']}", flush=True)
                if job.get("source") == "qwen" and job["dataset"] not in verified_qwen:
                    verify_qwen(args, config, job["dataset"])
                    verified_qwen.add(job["dataset"])
                reused = (correction_job(job, config) if job["kind"] == "correction" else
                          training_job(job) if job["kind"] == "training" else
                          summary_job(job, args, config, paper, datasets))
                record.update(status="completed", reused=reused)
                training.write_json(ledger / "status.json", state)
            state["stages"][phase] = "completed"
        state["status"] = "completed"
        training.write_json(ledger / "status.json", state)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        state["stages"][phase] = "failed"
        if state["jobs"] and state["jobs"][-1]["status"] == "running":
            state["jobs"][-1]["status"] = "failed"
        training.write_json(ledger / "status.json", state)
        raise
    return {"ledger": str(ledger), **state}


if __name__ == "__main__":
    run(parse_args())
