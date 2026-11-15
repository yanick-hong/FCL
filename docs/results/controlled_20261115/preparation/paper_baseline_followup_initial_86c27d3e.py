"""Finite seven-baseline followup; execution is explicitly gated on core success.

Default: write a plan, never train. --preflight checks selected datasets with
CUDA hidden. --execute waits for the complete controlled worker, then runs the
fixed main-table queue. A failed/partial core never becomes a completed queue.
"""
from __future__ import annotations

import argparse
import ast
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
    parser.add_argument("--resume", action="store_true", help="Pass strict fingerprint resume to the existing paper runner")
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
    if mode == "--execute" and args.resume:
        command.append("--resume")
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
    if os.name != "posix":
        raise RuntimeError("Training followup execution is supported on the authorized POSIX AutoDL host only")
    status_path = args.output_root / "followup_status.json"
    if status_path.exists() and not args.resume:
        raise FileExistsError("A followup status already exists; inspect it and explicitly use --resume")
    if status_path.exists():
        prior = read_json(status_path)
        for key in ("git_commit", "runner_source_sha256", "campaign_config_sha256", "paper_config_sha256", "wrapper_sha256"):
            if prior.get(key) != plan[key]:
                raise ValueError(f"Resume plan changed at {key}; do not mix outputs")
    status = {key: plan[key] for key in ("git_commit", "runner_source_sha256", "campaign_config_sha256", "paper_config_sha256", "wrapper_sha256")}
    status.update(status="pending_core", pid=os.getpid(), started_at=now(), jobs=plan["jobs"], training_started=False)
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
            assert_source(args, runner, plan)
            if core_state(plan)[0] != "completed":
                raise RuntimeError("Core completion evidence changed during baseline followup")
            dataset, source, method = (record[key] for key in ("dataset", "source", "method"))
            record.update(status="preflight", started_at=now(),
                          fixed_core_inputs=verify_fixed_core_inputs(args, dataset, source))
            status.update(current_job=record["id"], updated_at=now())
            save(status_path, status)
            stem = f"{index:03d}_{dataset}_{source}_{method}"
            record["preflight_log"] = str(args.output_root / "followup_logs" / f"{stem}_preflight.log")
            run_command(args, record["preflight_command"], Path(record["preflight_log"]), cpu=True)
            job = job_definition(args, runner, paper, dataset, source, method)
            expected, fingerprints = runner.job_fingerprint(job, plan["runner_source_sha256"], plan["git_dirty"])
            record.update(status="running", fingerprint=expected, fingerprints=fingerprints,
                          log=str(args.output_root / "followup_logs" / f"{stem}.log"))
            status["training_started"] = True
            save(status_path, status)
            run_command(args, record["command"], Path(record["log"]), cpu=False)
            assert_source(args, runner, plan)
            completed = runner.find_completed_job(args.output_root, runner.job_identity(job), expected)
            if completed is None:
                raise ValueError("Runner returned success without a completed exact-fingerprint result")
            metrics = read_json(completed / "metrics.json")
            if metric_accuracy(metrics) is None:
                raise ValueError("Completed result has no finite valid test accuracy")
            record.update(status="completed", completed_at=now(), run_dir=str(completed),
                          metrics_sha256=sha(completed / "metrics.json"))
            save(status_path, status)
        if len(status["jobs"]) != 84 or any(job["status"] != "completed" for job in status["jobs"]):
            raise ValueError("Finite queue did not complete every one of the 84 planned baseline jobs")
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
