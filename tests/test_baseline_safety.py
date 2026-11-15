"""Regression checks for baseline sampler determinism and cache contracts."""
from pathlib import Path

import pytest
import torch

from src.contrast import train_dld, train_l2b, cache_experiment


def _cache():
    labels = torch.arange(24) % 3
    return {
        "train_feats": torch.randn(24, 4), "train_labels": labels,
        "val_feats": torch.randn(6, 4), "val_labels": torch.arange(6) % 3,
        "test_feats": torch.randn(6, 4), "test_labels": torch.arange(6) % 3,
        "clip_probs_train": torch.nn.functional.one_hot(labels, 3).float(),
        "clip_probs_val": torch.nn.functional.one_hot(torch.arange(6) % 3, 3).float(),
        "clip_argmax_train": labels.clone(),
    }


def test_l2b_rejects_fabricated_source_probabilities(tmp_path: Path):
    bundle = _cache()
    bundle.pop("clip_probs_train")
    path = tmp_path / "cache.pt"
    torch.save(bundle, path)
    with pytest.raises(KeyError, match="source-label fields"):
        train_l2b.load_clip_cache_strict(str(path))


def test_baseline_samplers_replay_and_advance_by_epoch():
    labels = torch.arange(120) % 6
    for sampler_factory in (
        lambda: train_l2b.PKBatchSampler(labels, 3, 4, seed=9),
        lambda: train_dld.PKBatchSampler(labels, 3, 4, seed=9),
    ):
        first = sampler_factory()
        twin = sampler_factory()
        epoch0, epoch1 = list(first), list(first)
        twin0, twin1 = list(twin), list(twin)
        assert epoch0 == twin0
        assert epoch1 == twin1
        assert epoch0 != epoch1
        first.set_epoch(0)
        assert epoch0 == list(first)


@pytest.mark.parametrize("invalid", ["fractional_label", "fractional_flag", "nan_label", "split_length"])
def test_shared_cache_rejects_corrupt_labels_before_integer_cast(tmp_path: Path, invalid):
    bundle = _cache()
    obs = {"y_obs": bundle["train_labels"].clone(), "s": torch.zeros(24)}
    if invalid == "fractional_label":
        obs["y_obs"] = obs["y_obs"].float() + 0.5
    elif invalid == "fractional_flag":
        obs["s"][0] = 0.5
    elif invalid == "nan_label":
        obs["y_obs"] = obs["y_obs"].float()
        obs["y_obs"][0] = float("nan")
    else:
        bundle["test_labels"] = bundle["test_labels"][:-1]
    torch.save(bundle, tmp_path / "cache.pt")
    torch.save(obs, tmp_path / "obs.pt")
    with pytest.raises(ValueError):
        cache_experiment.load_cache_and_observed(str(tmp_path / "cache.pt"), str(tmp_path / "obs.pt"))
