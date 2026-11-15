import argparse
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.prepare_annotation_manifests import prepare
from scripts.run_qwen_campaign import appendix_prompt, run
from utils.annotation import qwen_batch


def inputs(tmp_path):
    cache = {"class_names": ["cat", "dog"], "sample_ids": {}, "sample_paths": {}}
    for split in ("train", "val", "test"):
        files = []
        for index in range(2):
            image = tmp_path / f"{split}-{index}.png"
            image.write_bytes(b"mock-image")
            files.append(str(image.resolve()))
        cache.update({f"{split}_feats": torch.eye(2), f"{split}_labels": torch.tensor([0, 1]),
                      f"clip_probs_{split}": torch.tensor([[0., 1.], [1., 0.]]),
                      f"clip_argmax_{split}": torch.tensor([1, 0])})
        cache["sample_ids"][split] = [f"{split}-0", f"{split}-1"]
        cache["sample_paths"][split] = files
    processed = tmp_path / "processed"
    processed.mkdir()
    path = processed / "toy_clip_vit-l-14_embeddings.pt"
    torch.save(cache, path)
    prepare("toy", path, tmp_path / "manifests" / "toy")
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/controlled_campaign.yaml").read_text())
    cfg["datasets"] = ["toy"]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    return argparse.Namespace(config=config, datasets=None, processed_root=processed,
        manifest_root=tmp_path / "manifests", output_root=tmp_path / "qwen", resume=False,
        limit=None, splits=None, model_path=None)


def test_qwen_campaign_records_predictions_accuracy_and_resume_provenance(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    class Predictor:
        metadata = {"resolved_commit": "fixed-mock-revision"}
        def __init__(self, *_):
            pass
        def __call__(self, paths, prompt, max_new_tokens, classes=None):
            assert "Reply with ONLY the integer id" in prompt
            return ["0", "1"]
    monkeypatch.setattr(qwen_batch, "BatchedQwenPredictor", Predictor)
    run(args)
    result = json.loads((args.output_root / "toy/label_accuracy.json").read_text())
    assert result["status"] == "completed"
    assert all(row["accuracy"] == 1 for row in result["splits"].values())
    final_cache = torch.load(result["derived_cache"], weights_only=True)
    assert all(torch.equal(final_cache[f"clip_argmax_{split}"], torch.tensor([0, 1]))
               for split in ("train", "val", "test"))
    manifest = (args.manifest_root / "toy/manifest.jsonl").read_text()
    assert "labels" not in manifest and "ground_truth" not in manifest
    args.resume = True
    run(args)
    final_cache["test_feats"][0, 0] = 17
    torch.save(final_cache, result["derived_cache"])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run(args)


def test_appendix_c_prompt_uses_stable_zero_based_menu():
    assert appendix_prompt("cifar100", ["aquarium_fish", "bear"]) == (
        "You are a classifier for the CIFAR-100. Given the image, choose exactly ONE class from the list below.\n"
        "0: aquarium fish\n1: bear\nReply with ONLY the integer id.")


@pytest.mark.parametrize("filename", ["manifest.jsonl", "classes.json"])
def test_campaign_rejects_manifest_or_class_reordering_before_inference(tmp_path, filename):
    args = inputs(tmp_path)
    path = args.manifest_root / "toy" / filename
    if filename.endswith("jsonl"):
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    else:
        path.write_text(json.dumps(["dog", "cat"]))
    with pytest.raises(ValueError, match="vocabulary changed"):
        run(args)


def test_duplicate_tiny_class_names_are_disambiguated_without_changing_ids(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    cache_path = args.processed_root / "toy_clip_vit-l-14_embeddings.pt"
    cache = torch.load(cache_path, weights_only=True)
    cache.update(class_names=["crane", "crane"], wnids=["n1", "n2"])
    torch.save(cache, cache_path)
    # A different frozen cache uses its own immutable manifest directory.
    args.manifest_root = tmp_path / "disambiguated_manifests"
    prepare("toy", cache_path, args.manifest_root / "toy")
    class Predictor:
        metadata = {"resolved_commit": "fixed"}
        def __init__(self, *_):
            pass
        def __call__(self, paths, prompt, max_new_tokens, classes=None):
            assert "0: crane (n1)" in prompt and "1: crane (n2)" in prompt
            return ["0", "1"]
    monkeypatch.setattr(qwen_batch, "BatchedQwenPredictor", Predictor)
    run(args)
    result = json.loads((args.output_root / "toy/label_accuracy.json").read_text())
    assert result["splits"]["train"]["accuracy"] == 1


def test_partial_resume_rejects_mixed_inference_environments(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    args.limit = 1
    class Predictor:
        packages = "first-version"
        def __init__(self, *_):
            self.metadata = {"resolved_commit": "fixed", "packages": self.packages}
        def __call__(self, paths, *_, classes=None):
            return ["0"] * len(paths)
    monkeypatch.setattr(qwen_batch, "BatchedQwenPredictor", Predictor)
    run(args)
    assert json.loads((args.output_root / "status.json").read_text())["status"] == "partial"
    Predictor.packages = "different-version"
    args.resume = True
    args.limit = None
    with pytest.raises(ValueError, match="environment changed"):
        run(args)
