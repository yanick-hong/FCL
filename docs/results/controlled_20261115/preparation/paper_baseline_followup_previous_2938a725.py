"""Finite seven-baseline followup; execution is explicitly gated on core success.

Default: write a plan, never train. --preflight checks selected datasets with
CUDA hidden. --execute waits for the complete controlled worker, then runs the
fixed main-table queue. A failed/partial core never becomes a completed queue.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

METHODS = ("zero_shot", "fl_c", "nl_c", "fs_c", "l2b", "dld", "csgn")
SOURCES = ("clip", "qwen")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def revision(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def environment(cpu=False):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path("/root/miniconda3/bin/python")
                        if Path("/root/miniconda3/bin/python").exists() else Path(sys.executable))
    parser.add_argument("--campaign-config", type=Path)
    parser.add_argument("--paper-config", type=Path)
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-datasets", nargs="+", default=["caltech101"])
    parser.add_argument("--poll-seconds", type=float, default=60.)
    parser.add_argument("--wait-timeout-hours", type=float, default=72.)
    parser.add_argument("--command-timeout-hours", type=float, default=24.)
    parser.add_argument("--resume", action="store_true", help="Reuse only wrapper-sealed, unchanged completed outputs")
    parser.add_argument("--_audit-run", type=Path, help=argparse.SUPPRESS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true", help="Default: write the finite plan only")
    modes.add_argument("--preflight", action="store_true", help="CPU-only, read-only input checks; never train")
    modes.add_argument("--execute", action="store_true", help="Wait for the complete core, then run the finite queue")
    return parser.parse_args(argv)


def configure(args):
    args.root = args.root.resolve()
    args.campaign_config = args.campaign_config or args.root / "configs/controlled_campaign.yaml"
    args.paper_config = args.paper_config or args.root / "configs/paper_reproduction.yaml"
    sys.path.insert(0, str(args.root))
    from scripts import run_paper_experiments as runner
    from scripts.summarize_paper_results import metric_accuracy
    campaign, paper = runner.load_yaml(args.campaign_config), runner.load_yaml(args.paper_config)
    datasets = campaign["main_table_datasets"]
    if (not isinstance(datasets, list) or len(datasets) != 6 or len(set(datasets)) != 6
            or "dtd" in datasets or not set(datasets).issubset(paper["datasets"])):
        raise ValueError("Followup requires the six distinct main_table_datasets from campaign config, without DTD")
    if set(METHODS) - set(runner.METHODS) or "fcl" in METHODS:
        raise ValueError("Paper runner methods changed; FCL must remain excluded")
    if not set(args.preflight_datasets).issubset(datasets):
        raise ValueError("Preflight datasets must belong to main_table_datasets")
    if len(set(args.preflight_datasets)) != len(args.preflight_datasets):
        raise ValueError("Preflight datasets must be distinct")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.poll_seconds, args.wait_timeout_hours, args.command_timeout_hours)):
        raise ValueError("Poll/timeout values must be finite and positive")
    if float(paper["protocol"]["rho"]) != .3:
        raise ValueError("The fixed prototype artifacts require rho=.3")
    args.core_root = args.core_root or args.root / "outputs" / campaign["campaign"]
    args.output_root = args.output_root or args.root / "outputs/paper_baseline_recheck"
    return runner, metric_accuracy, campaign, paper, datasets


def input_paths(args, dataset, source):
    cache = (args.root / "dataset/processed" / f"{dataset}_clip_vit-l-14_embeddings.pt" if source == "clip"
             else args.core_root / "qwen" / dataset / f"{dataset}_qwen_through-test.pt")
    observed = args.core_root / "corrections" / dataset / source / "prototype.pt"
    return cache, observed


def runner_arguments(args, dataset, source, methods, mode):
    cache, observed = input_paths(args, dataset, source)
    command = [str(args.python), str(args.root / "scripts/run_paper_experiments.py"),
               "--config", str(args.paper_config), "--suite", "main", "--datasets", dataset,
               "--methods", *methods, "--label-source", source, "--prompt", "C",
               "--seeds", "42", "--epochs", "100", "--num-workers", "0",
               "--cache-template", str(cache), "--obs-template", str(observed),
               "--output-root", str(args.output_root), mode]
    # The paper runner's resume checks inputs but does not seal outputs. Only
    # this wrapper may resume, after validating its complete output receipt.
    return command


def job_definition(args, runner, paper, dataset, source, method):
    cache, observed = input_paths(args, dataset, source)
    settings = argparse.Namespace(suite="main", datasets=[dataset], methods=[method], label_source=source,
        prompt="C", seeds=[42], epochs=100, cache_template=str(cache), obs_template=str(observed),
        output_root=args.output_root, num_workers=0, P=None, K=None, pairwise_tau=None)
    job = runner.build_jobs(paper, settings)[0]
    if job["cache"] != cache or job["obs"] != observed or job["epochs"] != 100 or job["seed"] != 42:
        raise ValueError("Existing runner template expansion does not preserve the fixed inputs/protocol")
    return job


def cli_options(path):
    options = set()
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            options.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                           and isinstance(arg.value, str) and arg.value.startswith("--"))
    return options


def make_plan(args, runner, campaign, paper, datasets):
    required = {"--preflight", "--execute", "--resume", "--cache-template", "--obs-template", "--epochs",
                "--num-workers", "--methods", "--datasets", "--label-source", "--seeds", "--output-root"}
    available = cli_options(args.root / "scripts/run_paper_experiments.py")
    if required - available:
        raise ValueError(f"Paper runner CLI lacks {sorted(required - available)}")
    jobs = []
    for dataset in datasets:
        for source in SOURCES:
            for method in METHODS:
                job = job_definition(args, runner, paper, dataset, source, method)
                inner = runner.command_for(job, paper)
                if inner:
                    options = cli_options(Path(inner[1]))
                    missing = {word for word in inner[2:] if word.startswith("--")} - options
                    if missing:
                        raise ValueError(f"{method} training CLI lacks {sorted(missing)}")
                jobs.append({"id": f"{dataset}/{source}/{method}", "dataset": dataset,
                    "source": source, "method": method, "status": "pending",
                    "cache": str(job["cache"]), "observed_labels": str(job["obs"]),
                    "preflight_command": runner_arguments(args, dataset, source, [method], "--preflight"),
                    "command": runner_arguments(args, dataset, source, [method], "--execute")})
    runner._SOURCE_FINGERPRINT = None
    source_sha, dirty = runner.source_fingerprint()
    return {"schema_version": 1, "created_at": now(), "root": str(args.root), "git_commit": revision(args.root),
        "runner_source_sha256": source_sha, "git_dirty": dirty, "wrapper_sha256": sha(Path(__file__)),
        "campaign_config_sha256": sha(args.campaign_config), "paper_config_sha256": sha(args.paper_config),
        "main_table_datasets": datasets, "methods": list(METHODS), "sources": list(SOURCES),
        "epochs": 100, "seed": 42, "resume_strict_fingerprints": args.resume,
        "resume_policy": "wrapper_sealed_output_receipts; paper runner --resume is never passed",
        "core_status": str(args.core_root / "worker_status.json"), "core_required_datasets": campaign["datasets"],
        "strict_paper_splits": False,
        "split_note": "Use the same recovered caches as core; preserve known paper-count/class discrepancies as audit warnings.",
        "exclusion": "FCL is already covered by the fixed-mask core's three seeds per source; no FCL rerun here.",
        "jobs": jobs}


def assert_source(args, runner, plan):
    runner._SOURCE_FINGERPRINT = None
    digest, dirty = runner.source_fingerprint()
    if (revision(args.root) != plan["git_commit"] or digest != plan["runner_source_sha256"]
            or dirty != plan["git_dirty"] or sha(args.campaign_config) != plan["campaign_config_sha256"]
            or sha(args.paper_config) != plan["paper_config_sha256"] or sha(Path(__file__)) != plan["wrapper_sha256"]):
        raise RuntimeError("HEAD/source/config/wrapper changed; preserve the pending queue and use a reviewed new plan")


def verify_fixed_core_inputs(args, dataset, source):
    """Bind baselines to the exact cache/prototype files used by all core seeds."""
    cache, observed = input_paths(args, dataset, source)
    expected = {"cache_sha256": sha(cache), "obs_sha256": sha(observed)}
    evidence = {}
    for seed in (42, 43, 44):
        path = args.core_root / "runs" / dataset / source / "prototype/fcl_auc" / f"seed-{seed}" / "metrics.json"
        result = read_json(path)
        identity = {"status": "completed", "dataset": dataset, "label_source": source,
                    "selection": "prototype", "objective": "fcl_auc", "seed": seed, "completed_epochs": 100}
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Core FCL evidence is incomplete/mismatched: {path}")
        if any(result.get("fingerprints", {}).get(key) != value for key, value in expected.items()):
            raise ValueError(f"Baseline cache/prototype differs from the completed core: {path}")
        accuracy = result.get("test_accuracy")
        if isinstance(accuracy, bool) or not isinstance(accuracy, (float, int)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Core FCL evidence has no finite test accuracy: {path}")
        evidence[str(seed)] = {"metrics_path": str(path), "metrics_sha256": sha(path)}
    return {**expected, "core_prototype_fcl_seeds": evidence}


def run_command(args, command, log_path, cpu):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "started_at": now(), "cpu_only": cpu}) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=args.root, env=environment(cpu), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        try:
            process.wait(timeout=args.command_timeout_hours * 3600)
        except subprocess.TimeoutExpired:
            # The paper runner launches a learner child. Terminate the entire
            # owned process group so a timeout cannot leave GPU training orphaned.
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.kill()  # Windows supports plan/preflight only.
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.write(json.dumps({"status": "timeout", "ended_at": now()}) + "\n")
            raise
        log.write(json.dumps({"returncode": process.returncode, "ended_at": now()}) + "\n")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def output_inventory(run_dir):
    """Seal every result byte, rather than trusting a mutable accuracy field."""
    root = Path(run_dir).resolve()
    items = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Result symlinks are not accepted: {path}")
        if path.is_file():
            items.append({"path": path.relative_to(root).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha(path)})
    if not items:
        raise ValueError("An empty result directory cannot be sealed")
    return items


def finite_fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite accuracy in [0,1]")
    return float(value)


def validate_output_cpu(descriptor):
    """Independently verify real learner formats; called only with CUDA hidden.

    The legacy CSVs round validation accuracy (6dp for L2B/DLD, 4dp for
    CSGN). L2B also overwrites checkpoint ties while its finalizer reports the
    first rounded maximum. These are recorded explicitly rather than guessing
    an epoch from a bare state_dict or rejecting a valid rounded history.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    run_dir, cache_path = Path(descriptor["run_dir"]), Path(descriptor["cache"])
    method, source = descriptor["method"], descriptor["source"]
    manifest = read_json(run_dir / "paper_run.json")
    if (manifest.get("status") != "completed" or manifest.get("measured") is not True
            or manifest.get("job_key") != descriptor["expected_identity"]
            or manifest.get("fingerprint") != descriptor["expected_fingerprint"]):
        raise ValueError("Output manifest is not the exact completed job")
    if sha(cache_path) != descriptor["cache_sha256"]:
        raise ValueError("Source cache changed before receipt audit")
    if descriptor.get("obs_sha256") is not None and sha(Path(descriptor["obs"])) != descriptor["obs_sha256"]:
        raise ValueError("Fixed prototype artifact changed before receipt audit")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    metrics = read_json(run_dir / "metrics.json")
    if metrics.get("method") != method:
        raise ValueError("Metrics method differs from the requested method")
    if metrics.get("source", source) != source or metrics.get("status", "completed") != "completed":
        raise ValueError("Metrics source/status differs from the completed job")
    proof = {"schema_version": 1, "method": method, "source": source,
             "device": "cpu", "torch": str(torch.__version__), "num_threads": 2,
             "cache_sha256": descriptor["cache_sha256"], "obs_sha256": descriptor.get("obs_sha256")}
    if method == "zero_shot":
        if metrics.get("measured") is not True or metrics.get("source") != source:
            raise ValueError("Zero-shot metrics lack explicit measured/source metadata")
        if source == "qwen":
            metadata = cache.get("annotation_metadata", {}).get("test", {})
            models = metadata.get("source_models", [])
            if (metadata.get("replaced_clip_fields") is not True or not isinstance(models, (list, tuple))
                    or not models or not all(str(m).lower().rsplit("/", 1)[-1].startswith("qwen") for m in models)):
                raise ValueError("Qwen zero-shot test predictions have no verified Qwen origin")
        test = metrics.get("metrics", {}).get("test", {})
        expected = finite_fraction(test.get("accuracy"), "zero-shot test accuracy")
        labels = cache["test_labels"].long()
        predictions = cache.get("clip_argmax_test")
        if predictions is None:
            predictions = cache["clip_probs_test"].argmax(-1)
        if predictions.shape != labels.shape or test.get("n") != len(labels):
            raise ValueError("Zero-shot test cardinality differs from the source cache")
        correct = int(predictions.eq(labels).sum())
        actual = float(predictions.eq(labels).float().mean())
        if correct != round(expected * len(labels)) or abs(actual - expected) > 1e-7:
            raise ValueError("Zero-shot accuracy differs from the actual test pseudo labels")
        proof.update(training_performed=False, completed_epochs=0,
                     zero_shot_exception="No training history or checkpoint exists for source-label evaluation",
                     test={"samples": len(labels), "correct": correct, "accuracy": actual})
        return proof

    if method not in METHODS or descriptor["epochs"] != 100:
        raise ValueError("Receipt requires a supported method and the fixed 100-epoch budget")
    relative_history = ("logs/val_metrics_dld_linear.csv" if method == "dld" else "logs/val_metrics.csv")
    with (run_dir / relative_history).open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    if len(history) != 100 or [float(row["epoch"]) for row in history] != list(range(1, 101)):
        raise ValueError("Training output must contain every epoch 1..100 exactly once")
    accuracy_key = "val_acc_head" if method == "dld" else "val_acc"
    loss_key = "val_ce_head" if method == "dld" else "val_loss"
    accuracies = [finite_fraction(float(row[accuracy_key]), "history validation accuracy") for row in history]
    if any(not math.isfinite(float(row[loss_key])) for row in history):
        raise ValueError("Nonfinite validation loss in training history")
    first_max = max(range(len(history)), key=lambda index: accuracies[index])
    selection = metrics.get("selection", metrics)
    if (selection.get("epochs_completed") != 100 or selection.get("best_epoch") != first_max + 1
            or abs(finite_fraction(selection.get("best_val_accuracy"), "selected validation accuracy")
                   - accuracies[first_max]) > 1e-10):
        raise ValueError("Reported selection/epochs disagree with the complete validation history")
    checkpoint_path = run_dir / "best.ckpt"
    if not checkpoint_path.is_file() or not checkpoint_path.stat().st_size:
        raise ValueError("Training output has no nonempty best checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("Unsupported best checkpoint structure")
    weights = [(state[w], state[b]) for w, b in
               (("linear.weight", "linear.bias"), ("fc.weight", "fc.bias"),
                ("classifier.weight", "classifier.bias")) if w in state and b in state]
    if not weights or any(not torch.equal(weights[0][i], pair[i]) for pair in weights[1:] for i in (0, 1)):
        raise ValueError("Checkpoint classifier aliases are missing or inconsistent")
    weight, bias = weights[0]
    if (weight.ndim != 2 or bias.shape != (weight.shape[0],)
            or weight.shape[1] != cache["train_feats"].shape[1]
            or not torch.isfinite(weight).all() or not torch.isfinite(bias).all()):
        raise ValueError("Invalid/nonfinite classifier checkpoint")
    if method in ("fl_c", "nl_c", "fs_c"):
        if not (run_dir / "last.ckpt").is_file():
            raise ValueError("Internal CE runner must retain its last checkpoint")
        last = torch.load(run_dir / "last.ckpt", map_location="cpu", weights_only=True)
        if not isinstance(last, dict) or any(key not in last or not torch.isfinite(last[key]).all()
                                           for key in ("fc.weight", "fc.bias")):
            raise ValueError("Internal CE runner must retain a valid last checkpoint")

    def evaluate(split):
        x = cache[split + "_feats"].float()
        # CSGN's original validation uses the already cached features as-is;
        # every finalizer normalizes test features and other methods normalize val.
        if split != "val" or method != "csgn":
            x = F.normalize(x, dim=-1)
        y = cache[split + "_labels"].long()
        if y.min() < 0 or y.max() >= weight.shape[0] or len(y) != len(x):
            raise ValueError("Classifier evaluation labels/cardinality are invalid")
        with torch.inference_mode():
            logits = F.linear(x, weight.float(), bias.float())
            correct = logits.argmax(-1).eq(y)
            return {"accuracy": float(correct.float().mean()), "correct": int(correct.sum()),
                    "samples": len(y), "nll": float(F.cross_entropy(logits, y))}

    validation, test_cpu = evaluate("val"), evaluate("test")
    tolerance = 5.1e-5 if method == "csgn" else 6e-7 if method in ("l2b", "dld") else 1e-7
    if abs(validation["accuracy"] - accuracies[first_max]) > tolerance:
        raise ValueError("Best checkpoint does not attain the maximum recorded validation accuracy")
    if method == "csgn":
        epoch = payload.get("epoch")
        if (not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 100
                or abs(finite_fraction(payload.get("val_acc"), "CSGN payload validation accuracy")
                       - validation["accuracy"]) > 1e-7
                or abs(accuracies[epoch - 1] - validation["accuracy"]) > tolerance):
            raise ValueError("CSGN checkpoint epoch/accuracy do not match its rounded history")
    test = metrics.get("test", {})
    expected_test = finite_fraction(test.get("accuracy"), "classifier test accuracy")
    if (test.get("valid_count") != test_cpu["samples"] or test.get("total_count") != test_cpu["samples"]
            or test_cpu["correct"] != round(expected_test * test_cpu["samples"])
            or abs(test_cpu["accuracy"] - expected_test) > 1e-7):
        raise ValueError("Classifier test metrics do not match the best checkpoint and complete test split")
    if test.get("nll") is not None and (not math.isfinite(float(test["nll"]))
                                        or abs(float(test["nll"]) - test_cpu["nll"]) > 1e-5):
        raise ValueError("Classifier test NLL differs from the best checkpoint")
    proof.update(training_performed=True, completed_epochs=100, history=relative_history,
                 history_first_max_epoch=first_max + 1, history_max_accuracy=accuracies[first_max],
                 validation_accuracy_tolerance=tolerance, checkpoint_validation_cpu=validation,
                 test=test_cpu, best_ckpt_sha256=sha(checkpoint_path),
                 checkpoint_policy=("last_true_max_tie; finalizer reports first rounded history maximum" if method == "l2b"
                                    else "first_true_max; CSGN CSV rounds to 4 decimals" if method == "csgn"
                                    else "first_true_max; legacy DLD CSV rounds to 6 decimals" if method == "dld"
                                    else "first_validation_max"))
    return proof


@contextlib.contextmanager
def queue_lock(output_root):
    """Kernel lock survives PID reuse and releases automatically on exit."""
    if os.name != "posix":
        raise RuntimeError("Followup execution requires POSIX flock")
    import fcntl
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".followup.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another followup process already holds this output queue lock") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "locked_at": now()}) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def matching_completed_runs(output_root, identity, fingerprint):
    matches = []
    for path in output_root.rglob("paper_run.json"):
        payload = read_json(path)
        if (payload.get("status") == "completed" and payload.get("job_key") == identity
                and payload.get("fingerprint") == fingerprint):
            matches.append(path.parent.resolve())
    return matches


def validate_resume_receipt(args, runner, plan, job, expected, record):
    if not record.get("receipt_path") or not record.get("receipt_sha256"):
        raise ValueError("Completed job has no wrapper output receipt; unsealed legacy results cannot be resumed")
    receipt_path = Path(record["receipt_path"]).resolve()
    if not receipt_path.is_relative_to((args.output_root / "followup_receipts").resolve()):
        raise ValueError("Receipt is outside this wrapper's output queue")
    if sha(receipt_path) != record["receipt_sha256"]:
        raise ValueError("Output receipt SHA256 changed")
    receipt = read_json(receipt_path)
    if (receipt.get("schema_version") != 1 or receipt.get("job_key") != runner.job_identity(job)
            or receipt.get("fingerprint") != expected or receipt.get("wrapper_sha256") != plan["wrapper_sha256"]
            or receipt.get("runner_source_sha256") != plan["runner_source_sha256"]
            or receipt.get("git_commit") != plan["git_commit"]):
        raise ValueError("Output receipt belongs to different inputs/source/protocol")
    run_dir = Path(receipt["run_dir"]).resolve()
    if not run_dir.is_relative_to(args.output_root.resolve()) or str(run_dir) != record.get("run_dir"):
        raise ValueError("Receipt run directory differs from the completed status")
    if receipt.get("files") != output_inventory(run_dir):
        raise ValueError("Sealed result files are missing, added or modified")
    if record.get("metrics_sha256") != sha(run_dir / "metrics.json"):
        raise ValueError("Completed status metrics SHA256 differs from the sealed output")
    proof = receipt.get("verification", {})
    if (proof.get("method") != job["method"] or proof.get("source") != job["source"]
            or proof.get("completed_epochs") != (0 if job["method"] == "zero_shot" else 100)):
        raise ValueError("Output receipt lacks the required epoch/source verification")
    return run_dir


def seal_output(args, runner, plan, job, expected, fingerprints, run_dir, stem):
    descriptor = {"run_dir": str(run_dir), "cache": str(job["cache"]), "obs": str(job["obs"]),
                  "method": job["method"], "source": job["source"], "epochs": job["epochs"],
                  "expected_identity": runner.job_identity(job), "expected_fingerprint": expected,
                  "cache_sha256": fingerprints["cache_sha256"], "obs_sha256": fingerprints["obs_sha256"]}
    request_path = args.output_root / "followup_audits" / f"{stem}_{run_dir.name}.json"
    save(request_path, descriptor)
    command = [str(args.python), str(Path(__file__).resolve()), "--_audit-run", str(request_path)]
    process = subprocess.run(command, cwd=args.root, env=environment(cpu=True), text=True,
                             encoding="utf-8", capture_output=True,
                             timeout=min(600., args.command_timeout_hours * 3600))
    log_path = request_path.with_suffix(".log")
    log_path.write_text(json.dumps({"command": command, "cpu_only": True}) + "\n"
                        + process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Independent CPU output audit failed; see {log_path}")
    verification = json.loads(process.stdout)
    receipt = {"schema_version": 1, "sealed_at": now(), "run_dir": str(run_dir),
               "job_key": runner.job_identity(job), "fingerprint": expected,
               "wrapper_sha256": plan["wrapper_sha256"], "git_commit": plan["git_commit"],
               "runner_source_sha256": plan["runner_source_sha256"], "verification": verification,
               "files": output_inventory(run_dir)}
    receipt_path = args.output_root / "followup_receipts" / job["dataset"] / job["source"] / job["method"] / f"{run_dir.name}.json"
    if receipt_path.exists():
        raise FileExistsError("Refusing to overwrite an existing sealed output receipt")
    save(receipt_path, receipt)
    return receipt_path


def preflight(args, runner, plan):
    destination = args.output_root / "preflight" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    status = {"mode": "cpu_preflight_only", "status": "running", "training_performed": False,
              "started_at": now(), "git_commit": plan["git_commit"], "steps": []}
    save(destination / "plan.json", plan)
    save(destination / "status.json", status)
    try:
        for dataset in args.preflight_datasets:
            for source in SOURCES:
                assert_source(args, runner, plan)
                command = runner_arguments(args, dataset, source, list(METHODS), "--preflight")
                record = {"dataset": dataset, "source": source, "status": "running", "command": command,
                          "log": str(destination / f"{dataset}_{source}.log"), "started_at": now(),
                          "fixed_core_inputs": verify_fixed_core_inputs(args, dataset, source)}
                status["steps"].append(record)
                save(destination / "status.json", status)
                run_command(args, command, Path(record["log"]), cpu=True)
                record.update(status="passed", completed_at=now())
                save(destination / "status.json", status)
        assert_source(args, runner, plan)
        status.update(status="passed", ended_at=now())
        save(destination / "status.json", status)
    except Exception as error:
        status.update(status="failed", error=repr(error), ended_at=now())
        save(destination / "status.json", status)
        raise
    print(json.dumps({"mode": status["mode"], "status": status["status"], "evidence": str(destination)}))
    return 0


def core_state(plan):
    path = Path(plan["core_status"])
    if not path.is_file():
        return "missing", None
    core = read_json(path)
    if core.get("status") == "failed":
        return "failed", core
    if core.get("status") != "completed":
        return "pending", core
    if core.get("git_commit") != plan["git_commit"]:
        return "inconsistent", core
    if any(core.get("datasets", {}).get(name, {}).get(key) != "completed"
           for name in plan["core_required_datasets"] for key in ("qwen", "controlled_suite")):
        return "inconsistent", core
    return "completed", core


def execute(args, runner, metric_accuracy, paper, plan):
    with queue_lock(args.output_root):
        return execute_locked(args, runner, metric_accuracy, paper, plan)


def execute_locked(args, runner, metric_accuracy, paper, plan):
    if os.name != "posix":
        raise RuntimeError("Training followup execution is supported on the authorized POSIX AutoDL host only")
    status_path = args.output_root / "followup_status.json"
    if status_path.exists() and not args.resume:
        raise FileExistsError("A followup status already exists; inspect it and explicitly use --resume")
    prior_jobs = {}
    if status_path.exists():
        prior = read_json(status_path)
        for key in ("git_commit", "runner_source_sha256", "campaign_config_sha256", "paper_config_sha256", "wrapper_sha256"):
            if prior.get(key) != plan[key]:
                raise ValueError(f"Resume plan changed at {key}; do not mix outputs")
        prior_jobs = {record["id"]: record for record in prior.get("jobs", [])}
        if len(prior_jobs) != 84 or set(prior_jobs) != {record["id"] for record in plan["jobs"]}:
            raise ValueError("Prior followup status does not describe the exact 84-job queue")
        save(args.output_root / "followup_attempts" /
             dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / "prior_status.json", prior)
    status = {key: plan[key] for key in ("git_commit", "runner_source_sha256", "campaign_config_sha256", "paper_config_sha256", "wrapper_sha256")}
    status.update(status="pending_core", pid=os.getpid(), started_at=now(), jobs=plan["jobs"], training_started=False)
    attempt = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    save(args.output_root / "followup_plan.json", plan)
    save(status_path, status)
    began = time.monotonic()
    while True:
        assert_source(args, runner, plan)
        gate, core = core_state(plan)
        status.update(core_gate=gate, checked_at=now(), core_status_sha256=sha(Path(plan["core_status"]))
                      if Path(plan["core_status"]).is_file() else None)
        if gate in ("failed", "inconsistent"):
            status.update(status=f"pending_core_{gate}", stop_reason="Core did not complete successfully; no baseline training launched")
            save(status_path, status)
            return 3
        if gate == "completed":
            break
        if time.monotonic() - began >= args.wait_timeout_hours * 3600:
            status.update(status="pending_core_timeout", stop_reason="Finite waiting limit reached; pending work preserved")
            save(status_path, status)
            return 2
        save(status_path, status)
        time.sleep(args.poll_seconds)
    status.update(status="running", core_completed_at=core.get("completed_at"))
    save(status_path, status)
    try:
        for index, record in enumerate(status["jobs"]):
            status["current_job"] = record["id"]
            assert_source(args, runner, plan)
            if core_state(plan)[0] != "completed":
                raise RuntimeError("Core completion evidence changed during baseline followup")
            dataset, source, method = (record[key] for key in ("dataset", "source", "method"))
            record.update(status="preflight", started_at=now(),
                          fixed_core_inputs=verify_fixed_core_inputs(args, dataset, source))
            job = job_definition(args, runner, paper, dataset, source, method)
            expected, fingerprints = runner.job_fingerprint(job, plan["runner_source_sha256"], plan["git_dirty"])
            previous = prior_jobs.get(record["id"], {})
            if previous.get("status") == "completed":
                completed = validate_resume_receipt(args, runner, plan, job, expected, previous)
                record.update(status="completed", resumed_from_receipt=True, fingerprint=expected,
                              run_dir=str(completed), receipt_path=previous["receipt_path"],
                              receipt_sha256=previous["receipt_sha256"],
                              metrics_sha256=sha(completed / "metrics.json"), completed_at=previous.get("completed_at"))
                save(status_path, status)
                continue
            if matching_completed_runs(args.output_root, runner.job_identity(job), expected):
                raise ValueError("A completed legacy/unsealed result has no valid wrapper receipt; refusing to reuse it")
            status.update(current_job=record["id"], updated_at=now())
            save(status_path, status)
            stem = f"{index:03d}_{dataset}_{source}_{method}"
            record["preflight_log"] = str(args.output_root / "followup_logs" / attempt / f"{stem}_preflight.log")
            run_command(args, record["preflight_command"], Path(record["preflight_log"]), cpu=True)
            assert_source(args, runner, plan)
            if core_state(plan)[0] != "completed":
                raise RuntimeError("Core completion evidence changed during CPU preflight")
            record.update(status="running", fingerprint=expected, fingerprints=fingerprints,
                          log=str(args.output_root / "followup_logs" / attempt / f"{stem}.log"))
            status["training_started"] = True
            save(status_path, status)
            before = {path.parent.resolve() for path in args.output_root.rglob("paper_run.json")}
            run_command(args, record["command"], Path(record["log"]), cpu=False)
            assert_source(args, runner, plan)
            candidates = [path for path in matching_completed_runs(args.output_root, runner.job_identity(job), expected)
                          if path not in before]
            if len(candidates) != 1:
                raise ValueError("Runner did not produce exactly one NEW completed exact-fingerprint result")
            completed = candidates[0]
            metrics = read_json(completed / "metrics.json")
            if metric_accuracy(metrics) is None:
                raise ValueError("Completed result has no finite valid test accuracy")
            receipt_path = seal_output(args, runner, plan, job, expected, fingerprints, completed, stem)
            record.update(status="completed", completed_at=now(), run_dir=str(completed),
                          metrics_sha256=sha(completed / "metrics.json"), receipt_path=str(receipt_path),
                          receipt_sha256=sha(receipt_path))
            save(status_path, status)
        if len(status["jobs"]) != 84 or any(job["status"] != "completed" for job in status["jobs"]):
            raise ValueError("Finite queue did not complete every one of the 84 planned baseline jobs")
        for record in status["jobs"]:
            job = job_definition(args, runner, paper, record["dataset"], record["source"], record["method"])
            expected, _ = runner.job_fingerprint(job, plan["runner_source_sha256"], plan["git_dirty"])
            validate_resume_receipt(args, runner, plan, job, expected, record)
        status.update(status="completed", completed_at=now())
        save(status_path, status)
        return 0
    except Exception as error:
        status.update(status="failed", error=repr(error), failed_at=now())
        if status.get("current_job"):
            active = next(job for job in status["jobs"] if job["id"] == status["current_job"])
            if active["status"] != "completed":
                active["status"] = "failed"
        save(status_path, status)
        raise


def main(argv=None):
    args = parse_args(argv)
    if args._audit_run:
        print(json.dumps(validate_output_cpu(read_json(args._audit_run)), allow_nan=False))
        return 0
    runner, metric_accuracy, campaign, paper, datasets = configure(args)
    plan = make_plan(args, runner, campaign, paper, datasets)
    if args.preflight:
        return preflight(args, runner, plan)
    if args.execute:
        return execute(args, runner, metric_accuracy, paper, plan)
    save(args.output_root / "followup_plan.json", plan)
    print(json.dumps({"mode": "plan_only", "jobs": len(plan["jobs"]), "training_performed": False,
                      "plan": str(args.output_root / "followup_plan.json")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
