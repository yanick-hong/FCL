import json
from pathlib import Path
import sys
import types

import pytest
import torch

from scripts import prepare_annotation_manifests as manifests


def legacy_caltech(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    root = raw / "caltech101"
    # The original Caltech loader numbers images numerically, not via
    # ImageFolder's lexicographic list of all filesystem paths.
    source = types.SimpleNamespace(root=str(root), categories=["cat", "dog"],
                                   y=[0, 0, 0, 1, 1, 1], index=[1, 2, 3, 1, 2, 3])
    for label, index in zip(source.y, source.index):
        image = root / "101_ObjectCategories" / source.categories[label] / f"image_{index:04d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"identity-test-image")
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace(__version__="mock",
        datasets=types.SimpleNamespace(Caltech101=lambda **_: source)))
    monkeypatch.setattr(manifests, "RAW_ROOT", raw)
    cache = {"class_names": ["cat", "dog"]}
    for split, indices in zip(("train", "val", "test"), ([1, 4], [0, 3], [2, 5])):
        cache.update({f"{split}_indices": torch.tensor(indices), f"{split}_labels": torch.tensor([0, 1]),
                      f"{split}_feats": torch.eye(2), f"clip_probs_{split}": torch.eye(2)})
    path = tmp_path / "caltech.pt"
    torch.save(cache, path)
    return path, cache


def test_recover_legacy_caltech_order_from_original_loader_and_frozen_indices(tmp_path, monkeypatch):
    path, _ = legacy_caltech(tmp_path, monkeypatch)
    output = tmp_path / "manifest"
    metadata = manifests.prepare("caltech101", path, output)
    rows = [json.loads(row) for row in (output / "manifest.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows[:2]] == ["caltech101/00001", "caltech101/00004"]
    assert Path(rows[0]["path"]).name == "image_0002.jpg"
    assert metadata["legacy_cache_order_verified"]
    assert metadata["identity_recovery"]["class_order_and_all_labels_verified"]
    assert "sample_ids" not in torch.load(path, weights_only=True)  # Frozen cache remains unchanged.


@pytest.mark.parametrize("indices", [[0, 1], [-1, 4], [1, 1], [1+9j, 4+0j]])
def test_recovery_rejects_misaligned_or_invalid_frozen_indices(tmp_path, monkeypatch, indices):
    path, cache = legacy_caltech(tmp_path, monkeypatch)
    cache["train_indices"] = torch.tensor(indices)
    torch.save(cache, path)
    with pytest.raises(ValueError):
        manifests.prepare("caltech101", path, tmp_path / "manifest")


def test_recovery_cannot_hide_conflicting_flat_paths_or_fractional_labels(tmp_path, monkeypatch):
    path, cache = legacy_caltech(tmp_path, monkeypatch)
    recovered, _, _ = manifests.recover_indexed_images("caltech101", cache, manifests.RAW_ROOT)
    cache["train_paths"] = list(reversed(recovered["train"]))
    torch.save(cache, path)
    with pytest.raises(ValueError, match="alias disagrees"):
        manifests.prepare("caltech101", path, tmp_path / "manifest")
    cache.pop("train_paths")
    cache["train_labels"] = torch.tensor([.4, 1.])
    torch.save(cache, path)
    with pytest.raises(ValueError, match="ground-truth labels"):
        manifests.prepare("caltech101", path, tmp_path / "manifest")
