"""Exercise the real annotation exporter and paper runner across split overlays."""
import argparse
import json

import pytest
import torch

from scripts.run_paper_experiments import inspect_input, run_ce_or_zeroshot
from src.utils.annotation.cli import export_cache
from src.utils.annotation.schema import Manifest, make_record
from src.utils.annotation.store import AnnotationStore


def test_exported_qwen_splits_are_used_without_clip_test_fallback(tmp_path):
    classes = ["cat", "dog"]
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(classes), encoding="utf-8")
    samples = [{"sample_id": f"{split}-{i}", "path": f"{split}/{i}.jpg", "split": split}
               for split in ("train", "val", "test") for i in range(2)]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(row) for row in samples), encoding="utf-8")
    original = {}
    for split in ("train", "val", "test"):
        original.update({f"{split}_feats": torch.zeros(2, 768),
                         f"{split}_labels": torch.tensor([0, 1]),
                         f"{split}_sample_ids": [f"{split}-0", f"{split}-1"],
                         f"clip_probs_{split}": torch.tensor([[0., 1.], [1., 0.]]),
                         f"clip_argmax_{split}": torch.tensor([1, 0])})
    input_cache = tmp_path / "clip.pt"
    torch.save(original, input_cache)

    def overlay(split, cache, output_name):
        rows = [row for row in samples if row["split"] == split]
        annotations = tmp_path / f"{split}.jsonl"
        store = AnnotationStore(annotations, Manifest(rows), classes, "Qwen/Qwen2.5-VL-7B-Instruct",
                                {"prompt_variant": "C", "prompt_sha256": "known-prompt"})
        # Reversed annotation order must still match the cache's stable IDs.
        for i in (1, 0):
            store.append(make_record(rows[i], i, classes, "Qwen/Qwen2.5-VL-7B-Instruct"))
        output = tmp_path / output_name
        export_cache(argparse.Namespace(manifest=str(manifest_path), split=split, classes=str(classes_path),
            annotations=str(annotations), cache=str(cache), output=str(output),
            confirm_cache_manifest_order=False, replace_clip_fields=True))
        return output

    train_cache = overlay("train", input_cache, "qwen-train.pt")
    cfg = {"datasets": {"toy": {"paper_train_samples": 2, "paper_classes": 2}}}
    job = {"dataset": "toy", "cache": train_cache, "obs": tmp_path / "missing-observed.pt",
           "source": "qwen", "method": "fcl", "prompt": "C", "rho": .3, "run_dir": tmp_path}
    cache, _ = inspect_input(job, cfg, strict=True)
    assert cache["clip_argmax_train"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="Selected CLIP source contains Qwen"):
        inspect_input({**job, "source": "clip"}, cfg, strict=True)
    # FCL needs source priors only in train, never val/test pseudo-labels.
    train_only = dict(cache)
    for split in ("val", "test"):
        del train_only[f"clip_probs_{split}"]
        del train_only[f"clip_argmax_{split}"]
    train_only_path = tmp_path / "qwen-train-only.pt"
    torch.save(train_only, train_only_path)
    inspect_input({**job, "cache": train_only_path}, cfg, strict=True)

    zero_shot = {**job, "method": "zero_shot"}
    with pytest.raises(ValueError, match="test.*CLIP fallback is forbidden"):
        inspect_input(zero_shot, cfg, strict=True)

    # A separately exported test overlay makes the Qwen zero-shot run valid.
    test_cache = overlay("test", train_cache, "qwen-train-test.pt")
    zero_shot["cache"] = test_cache
    cache, _ = inspect_input(zero_shot, cfg, strict=True)
    with pytest.raises(ValueError, match="prompt metadata C does not match requested A"):
        inspect_input({**zero_shot, "prompt": "A"}, cfg, strict=True)
    run_ce_or_zeroshot(zero_shot, cfg, cache)
    result = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert result["source"] == "qwen"
    assert result["metrics"]["test"] == {"accuracy": 1.0, "n": 2}
    assert "val" not in result["metrics"]  # untouched CLIP predictions are never reported as Qwen
