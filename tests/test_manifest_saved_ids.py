"""Recover Tiny/DTD paths from frozen identities without touching features."""
import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import prepare_annotation_manifests as manifests


def cache_splits(classes, labels):
    cache = {"sample_ids": {}}
    for split, targets in labels.items():
        truth = torch.tensor(targets)
        cache.update({f"{split}_labels": truth, f"{split}_feats": torch.eye(len(targets)),
                      f"clip_probs_{split}": torch.nn.functional.one_hot(truth, len(classes)).float()})
    return cache


def tiny_inputs(tmp_path):
    raw = tmp_path / "raw"
    base = raw / "tiny-imagenet-200"
    base.mkdir(parents=True)
    (base / "wnids.txt").write_text("n1\nn2\n")
    annotations = base / "val/val_annotations.txt"
    annotations.parent.mkdir(parents=True)
    annotations.write_text("val_001.JPEG\tn2\t0\t0\t1\t1\nval_002.JPEG\tn1\t0\t0\t1\t1\n")
    cache = cache_splits(["cat", "dog"], {"train": [1, 0], "val": [0, 1], "test": [0, 1]})
    cache.update(classnames=["cat", "dog"], wnids=["n1", "n2"])
    cache["sample_ids"] = {
        "train": ["train/n2/images/n2_t.JPEG", "train/n1/images/n1_t.JPEG"],
        "val": ["train/n1/images/n1_v.JPEG", "train/n2/images/n2_v.JPEG"],
        # Identity order is intentionally different from the annotations file.
        "test": ["val/images/val_002.JPEG", "val/images/val_001.JPEG"]}
    for ids in cache["sample_ids"].values():
        for value in ids:
            image = base / value
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"frozen-image")
    path = tmp_path / "tiny_cache.pt"
    torch.save(cache, path)
    return raw, path, cache


def dtd_inputs(tmp_path):
    raw = tmp_path / "raw"
    base = raw / "dtd"
    (base / "labels").mkdir(parents=True)
    entries = {"train": ["b/b_1.jpg", "a/a_1.jpg"], "val": ["a/a_2.jpg", "b/b_2.jpg"],
               "test": ["b/b_3.jpg", "a/a_3.jpg"]}
    cache = cache_splits(["a", "b"], {"train": [1, 0], "val": [0, 1], "test": [1, 0]})
    cache["meta"] = {"classes": ["a", "b"], "class_to_idx": {"a": 0, "b": 1}, "partition": 1}
    for split, rows in entries.items():
        (base / "labels" / f"{split}1.txt").write_text("\n".join(rows) + "\n")
        ids = []
        for row in rows:
            image = base / "images" / row
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"frozen-image")
            ids.append(str(image.resolve()))
        cache["sample_ids"][split] = ids
    path = tmp_path / "dtd_cache.pt"
    torch.save(cache, path)
    return raw, path, cache


@pytest.mark.parametrize("dataset,fixture", [("tinyimagenet200", tiny_inputs), ("dtd", dtd_inputs)])
def test_saved_ids_recover_exact_order_and_leave_frozen_cache_unchanged(tmp_path, dataset, fixture):
    raw, path, cache = fixture(tmp_path)
    before = manifests.sha256_file(path)
    output = tmp_path / "manifests"
    metadata = manifests.prepare(dataset, path, output, raw_root=raw)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [identity for split in ("train", "val", "test")
                                                  for identity in cache["sample_ids"][split]]
    assert metadata["identity_recovery"]["class_order_and_all_labels_verified"]
    assert metadata["identity_recovery"]["historical_features_reencoded"] is False
    assert metadata["legacy_cache_order_verified"] is False
    assert metadata["identity_recovery"]["source_files_sha256"]
    assert manifests.sha256_file(path) == before
    assert "sample_paths" not in torch.load(path, weights_only=True)
    assert all(Path(row["path"]).is_file() for row in rows)


@pytest.mark.parametrize("fault", ["wrong_train_label", "wrong_test_label", "unsafe_path", "wnid_order", "alias_conflict"])
def test_tiny_recovery_rejects_wrong_labels_paths_and_aliases(tmp_path, fault):
    raw, path, cache = tiny_inputs(tmp_path)
    if fault == "wrong_train_label":
        cache["train_labels"][0] = 0
    elif fault == "wrong_test_label":
        cache["test_labels"][0] = 1
    elif fault == "unsafe_path":
        cache["sample_ids"]["train"][0] = "../train/n2/images/n2_t.JPEG"
    elif fault == "wnid_order":
        cache["wnids"] = ["n2", "n1"]
    else:
        base = raw / "tiny-imagenet-200"
        cache["train_paths"] = [str(base / value) for value in reversed(cache["sample_ids"]["train"])]
    torch.save(cache, path)
    with pytest.raises(ValueError):
        manifests.prepare("tinyimagenet200", path, tmp_path / "output", raw_root=raw)


@pytest.mark.parametrize("fault", ["order", "label", "partition", "escape", "alias_conflict"])
def test_dtd_recovery_checks_partition_order_labels_and_absolute_root(tmp_path, fault):
    raw, path, cache = dtd_inputs(tmp_path)
    if fault == "order":
        cache["sample_ids"]["train"].reverse()
    elif fault == "label":
        cache["train_labels"][0] = 0
    elif fault == "partition":
        cache["meta"]["partition"] = 2
    elif fault == "escape":
        cache["sample_ids"]["train"][0] = str(tmp_path / "unrelated.jpg")
    else:
        cache["sample_ids_train"] = list(reversed(cache["sample_ids"]["train"]))
    torch.save(cache, path)
    with pytest.raises(ValueError):
        manifests.prepare("dtd", path, tmp_path / "output", raw_root=raw)
