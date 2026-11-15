from __future__ import annotations

import argparse
import json
from pathlib import Path
import pytest

from scripts.run_paper_experiments import (
    build_jobs,
    command_for,
    find_completed_job,
    job_fingerprint,
    job_identity,
    load_yaml,
)
from scripts.summarize_paper_results import metric_accuracy


def runner_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        suite="main", datasets=["cifar100"], methods=["fcl"], label_source="clip",
        prompt="C", seeds=[42], epochs=3, cache_template=None, obs_template=None,
        output_root=tmp_path, prepare_labels=False, preflight=False,
        strict_paper_splits=False, execute=False, dry_run=True, continue_on_error=False,
        num_workers=0, P=4, K=5, pairwise_tau=0.7, resume=False,
    )


def test_build_jobs_propagates_overrides(tmp_path: Path):
    cfg = load_yaml(Path(__file__).parents[1] / "configs" / "paper_reproduction.yaml")
    job = build_jobs(cfg, runner_args(tmp_path))[0]
    command = command_for(job, cfg)
    assert job["P"] == 4 and job["K"] == 5
    assert "--tau" in command and command[command.index("--tau") + 1] == "0.7"
    assert "--num_workers" in command and command[command.index("--num_workers") + 1] == "0"


def test_build_jobs_rejects_more_classes_than_dataset(tmp_path: Path):
    cfg = load_yaml(Path(__file__).parents[1] / "configs" / "paper_reproduction.yaml")
    args = runner_args(tmp_path)
    args.datasets = ["cifar100"]
    args.P = 101
    with pytest.raises(ValueError, match="cannot draw more than 100"):
        build_jobs(cfg, args)


def test_resume_matches_job_and_fingerprint(tmp_path: Path):
    cfg = load_yaml(Path(__file__).parents[1] / "configs" / "paper_reproduction.yaml")
    job = build_jobs(cfg, runner_args(tmp_path))[0]
    fingerprint, _ = job_fingerprint(job, "source-sha")
    identity = job_identity(job)
    run_dir = tmp_path / "old"
    run_dir.mkdir()
    (run_dir / "paper_run.json").write_text(json.dumps({
        "status": "completed", "job_key": identity, "fingerprint": fingerprint,
    }), encoding="utf-8")
    assert find_completed_job(tmp_path, identity, fingerprint) == run_dir


def test_summary_extracts_supported_accuracy_shapes():
    assert metric_accuracy({"test_accuracy": 0.81}) == 0.81
    assert metric_accuracy({"test": {"accuracy": 81.0}}) == 0.81
    assert metric_accuracy({"metrics": {"test": {"accuracy": 0.79}}}) == 0.79
    assert metric_accuracy({}) is None
