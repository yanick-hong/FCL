"""Ordered-suite checks using real tiny correction/training artifacts."""
import argparse
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compare_label_corrections as correction
from scripts import run_controlled_suite as suite
from scripts import run_controlled_training as training
from scripts.prepare_annotation_manifests import prepare
from scripts import run_qwen_campaign as qwen
from utils.annotation import qwen_batch


@pytest.fixture
def tiny_campaign(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    processed = tmp_path / "processed"
    processed.mkdir()
    cache = {"class_names": ["cat", "dog"], "sample_ids": {}, "sample_paths": {}}
    generator = torch.Generator().manual_seed(5)
    for split, count in (("train", 12), ("val", 4), ("test", 6)):
        truth = torch.arange(count) % 2
        source_labels = truth.clone()
        source_labels[::3] = 1 - source_labels[::3]
        features = F.one_hot(truth, 2).float() + .03 * torch.randn(count, 2, generator=generator)
        cache.update({f"{split}_feats": features, f"{split}_labels": truth,
                      f"clip_probs_{split}": .1 + .8 * F.one_hot(source_labels, 2).float(),
                      f"clip_argmax_{split}": source_labels})
        cache["sample_ids"][split] = [f"{split}-{index}" for index in range(count)]
        paths = []
        for index in range(count):
            image = tmp_path / f"{split}-{index}.png"
            image.write_bytes(b"mocked-image")
            paths.append(str(image.resolve()))
        cache["sample_paths"][split] = paths
    cache_path = processed / "caltech101_clip_vit-l-14_embeddings.pt"
    torch.save(cache, cache_path)
    manifests = tmp_path / "manifests"
    prepare("caltech101", cache_path, manifests / "caltech101")
    config = yaml.safe_load((ROOT / "configs/controlled_campaign.yaml").read_text(encoding="utf-8"))
    config.update(datasets=["caltech101"], campaign="tiny_controlled")
    config["training"]["epochs"] = 3
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    paper_path = tmp_path / "paper.yaml"
    paper_path.write_text(yaml.safe_dump({"datasets": {"caltech101": {"P": 2, "K": 2}}}), encoding="utf-8")
    output = tmp_path / "campaign"
    class Predictor:
        def __init__(self, *_):
            self.metadata = {"resolved_commit": "fixed-test-commit", "packages": {"mocked": "1"}}
        def __call__(self, paths, *_):
            return [str(int(Path(path).stem.rsplit("-", 1)[1]) % 2) for path in paths]
    monkeypatch.setattr(qwen_batch, "BatchedQwenPredictor", Predictor)
    qwen_args = argparse.Namespace(config=config_path, datasets=None, splits=None, processed_root=processed,
        manifest_root=manifests, output_root=output / "qwen", resume=False, model_path=None, limit=None)
    qwen.run(qwen_args)
    dispatched = []
    def dispatch(command):
        dispatched.append(command)
        if Path(command[1]).name == "compare_label_corrections.py":
            assert correction.main(command[2:]) == 0
        elif Path(command[1]).name == "run_controlled_training.py":
            training.run(training.parse_args(command[2:]))
        else:
            raise AssertionError(command)
    monkeypatch.setattr(suite, "execute_command", dispatch)
    args = suite.parse_args(["--config", str(config_path), "--paper-config", str(paper_path),
        "--processed-root", str(processed), "--manifest-root", str(manifests),
        "--output-root", str(output), "--device", "cpu", "--execute"])
    yield args, dispatched
    torch.set_num_threads(previous_threads)
    torch.use_deterministic_algorithms(previous_deterministic)


def test_dry_run_has_ordered_stages_and_dtd_recovery_pk_without_writes(tmp_path):
    output = tmp_path / "planned"
    args = suite.parse_args(["--datasets", "caltech101", "dtd", "--output-root", str(output), "--dry-run"])
    plan = suite.run(args)
    assert plan["phases"] == list(suite.PHASES)
    assert not output.exists()
    assert len(plan["jobs"]) == 29
    phase_positions = [suite.PHASES.index(job["phase"]) for job in plan["jobs"]]
    assert phase_positions == sorted(phase_positions)
    dtd = next(job for job in plan["jobs"] if job.get("dataset") == "dtd" and job["kind"] == "training")
    assert dtd["command"][dtd["command"].index("--P") + 1] == "16"
    assert dtd["command"][dtd["command"].index("--K") + 1] == "16"


def test_missing_qwen_fails_before_any_correction_and_records_stage(tmp_path):
    output = tmp_path / "campaign"
    args = suite.parse_args(["--datasets", "caltech101", "--output-root", str(output),
                             "--processed-root", str(tmp_path / "missing"), "--phases", "corrections", "--execute"])
    with pytest.raises(FileNotFoundError):
        suite.run(args)
    assert not (output / "corrections").exists()
    state = json.loads(next((output / "orchestration").glob("*/status.json")).read_text())
    assert state["status"] == "failed"
    assert state["stages"]["corrections"] == "failed"


def test_full_suite_reuses_verified_jobs_and_rejects_changed_protocol(tiny_campaign):
    args, dispatched = tiny_campaign
    result = suite.run(args)
    assert result["status"] == "completed"
    assert len(dispatched) == 14  # 2 correction jobs + 12 training seeds.
    assert all(not job["reused"] for job in result["jobs"])
    summary = json.loads((args.output_root / "summaries/caltech101/summary.json").read_text())
    assert len(summary["groups"]) == 4
    assert len(summary["paired_objective_differences"]) == 1
    random_group = next(group for group in summary["groups"] if group["selection"] == "random_global_exact")
    assert random_group["selection_seed"] == 42
    assert random_group["seeds"] == [42, 43, 44]
    reused = suite.run(args)
    assert all(job["reused"] for job in reused["jobs"])
    assert len(dispatched) == 14
    path = args.output_root / "runs/caltech101/clip/random_global_exact/fcl_auc/seed-42/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["fingerprints"]["protocol_sha256"] = "changed"
    path.write_text(json.dumps(metrics))
    tampered = path.read_bytes()
    with pytest.raises(ValueError, match="fingerprints differ"):
        suite.run(args)
    assert path.read_bytes() == tampered
    assert len(dispatched) == 14


def test_partial_training_is_preserved_and_child_failure_stops_suite(tiny_campaign, monkeypatch):
    args, dispatched = tiny_campaign
    args.phases = ["corrections"]
    suite.run(args)
    incomplete = args.output_root / "runs/caltech101/clip/random_global_exact/fcl_auc/seed-42"
    incomplete.mkdir(parents=True)
    marker = incomplete / "unfinished.txt"
    marker.write_text("preserve")
    args.phases = ["train_random_auc"]
    with pytest.raises(FileExistsError, match="Incomplete training output preserved"):
        suite.run(args)
    assert marker.read_text() == "preserve"
    assert len(dispatched) == 2
    # A different phase starts no later jobs after a child process failure.
    import subprocess
    def fail(command):
        raise subprocess.CalledProcessError(7, command)
    monkeypatch.setattr(suite, "execute_command", fail)
    args.phases = ["train_soft_ce", "train_core"]
    with pytest.raises(subprocess.CalledProcessError):
        suite.run(args)
    latest = sorted((args.output_root / "orchestration").glob("*/status.json"))[-1]
    state = json.loads(latest.read_text())
    assert state["status"] == "failed"
    assert state["stages"]["train_core"] == "pending"
    assert len(state["jobs"]) == 1


def test_qwen_artifact_tampering_blocks_correction_without_clip_fallback(tiny_campaign):
    args, dispatched = tiny_campaign
    overlay = args.output_root / "qwen/caltech101/caltech101_qwen_through-val.pt"
    overlay.write_bytes(overlay.read_bytes() + b"changed")
    args.phases = ["corrections"]
    with pytest.raises(ValueError, match="overlay output fingerprint mismatch"):
        suite.run(args)
    assert not dispatched
    assert not (args.output_root / "corrections").exists()
