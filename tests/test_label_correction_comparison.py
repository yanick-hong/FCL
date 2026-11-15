"""Correction masks must preserve supervision boundaries and real budgets."""
import copy
import json

import pytest
import torch
import torch.nn.functional as F

from scripts.compare_label_corrections import (
    compare_cache, file_sha256, random_mask, stable_sample_order, tensor_sha256,
)


def make_cache():
    pseudo = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    cache = {"train_feats": torch.tensor([[1., 0], [1., 0], [1., 0], [-1., 0],
                                          [0., 1], [0., 1], [0., 1], [0., -1],
                                          [1., 1], [-1., -1]]),
             "train_labels": torch.tensor([0, 0, 0, 2, 1, 1, 1, 0, 1, 0]),
             "clip_probs_train": F.one_hot(pseudo, 3).float(), "clip_argmax_train": pseudo,
             "sample_ids": {"train": [f"row-{i}" for i in range(10)]}}
    for split, predicted, truth in (("val", [0, 1], [0, 2]), ("test", [2, 1], [2, 1])):
        labels = torch.tensor(predicted)
        cache.update({f"{split}_feats": torch.tensor([[1., 0], [0., 1]]),
                      f"{split}_labels": torch.tensor(truth),
                      f"clip_probs_{split}": F.one_hot(labels, 3).float(), f"clip_argmax_{split}": labels})
    return cache


def test_fixed_masks_obey_oracle_boundary_and_keep_distinct_budgets(tmp_path):
    cache = make_cache()
    path = tmp_path / "features.pt"
    torch.save(cache, path)
    original_sha = file_sha256(path)
    output = tmp_path / "comparison"
    report = compare_cache(path, "clip", "toy", output)
    assert file_sha256(path) == original_sha
    assert report["cache_sha256"] == original_sha
    assert report["source_metrics"]["train"]["accuracy"] == pytest.approx(.6)
    assert report["source_metrics"]["val"]["accuracy"] == .5
    assert report["source_metrics"]["test"]["accuracy"] == 1.
    assert report["prototype_actual_budget"] == 5  # ceil(4*.3)+ceil(4*.3)+ceil(2*.3)
    assert report["random_global_exact_budget"] == 3
    prototype = torch.load(output / "prototype.pt", weights_only=True)
    expected_rows = torch.tensor([0, 3, 4, 7, 8])
    assert torch.equal(torch.where(prototype["selected_mask"])[0], expected_rows)
    assert prototype["meta"]["selected_error_precision"] == pytest.approx(3 / 5)
    assert prototype["meta"]["selected_error_recall"] == pytest.approx(3 / 4)
    assert prototype["meta"]["corrected_label_accuracy"] == pytest.approx(.9)
    assert prototype["meta"]["selection_seed"] is None
    assert prototype["meta"]["predicted_class_budgets"]["0"]["selected"] == 2
    assert prototype["train_sample_ids"] == cache["sample_ids"]["train"]
    assert len(report["conditions"]) == 7
    for condition in report["conditions"]:
        observed = torch.load(condition["observed_labels"], weights_only=True)
        selected = observed["selected_mask"]
        assert torch.equal(observed["s"].eq(0), selected)
        assert torch.equal(observed["y_pseudo"], cache["clip_argmax_train"])
        assert torch.equal(observed["y_obs"][selected], cache["train_labels"][selected])
        assert torch.equal(observed["y_obs"][~selected], observed["y_pseudo"][~selected])
        assert observed["meta"]["mask_sha256"] == tensor_sha256(selected)
        assert observed["meta"]["y_obs_sha256"] == tensor_sha256(observed["y_obs"])
        assert observed["meta"]["schema_version"] == 1
        if condition["method"] == "random_global_exact":
            assert int(selected.sum()) == 3
        elif condition["method"] == "random_matched_prototype":
            assert int(selected.sum()) == 5
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["conditions"] == report["conditions"]
    assert (output / "report.csv").exists() and (output / "sample_order.json").exists()
    replay = compare_cache(path, "clip", "toy", tmp_path / "replay")
    assert [row["mask_sha256"] for row in report["conditions"]] == [row["mask_sha256"] for row in replay["conditions"]]
    with pytest.raises(FileExistsError):
        compare_cache(path, "clip", "toy", output)


def test_qwen_accuracy_is_split_specific_and_never_uses_clip_fallback(tmp_path):
    cache = make_cache()
    cache["annotation_metadata"] = {"train": {"replaced_clip_fields": True,
        "source_models": ["Qwen/Qwen2.5-VL-7B-Instruct"], "prompt_variant": "C"}}
    path = tmp_path / "qwen.pt"
    torch.save(cache, path)
    report = compare_cache(path, "qwen", "toy", tmp_path / "qwen-output", seeds=[42])
    assert set(report["source_metrics"]) == {"train"}
    assert set(report["omitted_source_splits"]) == {"val", "test"}
    with pytest.raises(ValueError, match="verified clip.*fallback is forbidden"):
        compare_cache(path, "clip", "toy", tmp_path / "wrong-source")
    with pytest.raises(ValueError, match="prompt C differs from requested A"):
        compare_cache(path, "qwen", "toy", tmp_path / "wrong-prompt", prompt="A")
    del cache["annotation_metadata"]
    torch.save(cache, path)
    with pytest.raises(ValueError, match="verified qwen.*fallback is forbidden"):
        compare_cache(path, "qwen", "toy", tmp_path / "missing-source")
    assert not (tmp_path / "missing-source").exists()


def test_global_budget_rounds_half_up_and_selection_rng_is_local(tmp_path):
    cache = make_cache()
    for key in ("train_feats", "train_labels", "clip_probs_train", "clip_argmax_train"):
        cache[key] = cache[key][:5]
    cache["sample_ids"]["train"] = cache["sample_ids"]["train"][:5]
    path = tmp_path / "five.pt"
    torch.save(cache, path)
    state = torch.random.get_rng_state()
    report = compare_cache(path, "clip", "toy", tmp_path / "five", seeds=[42])
    assert report["random_global_exact_budget"] == 2  # floor(1.5+.5), not bankers' rounding
    assert report["global_requested_ratio_is_exactly_representable"] is False
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(random_mask(10, 3, 42), random_mask(10, 3, 42))
    assert not torch.equal(random_mask(100, 30, 42), random_mask(100, 30, 43))


def test_invalid_probabilities_and_ambiguous_sample_order_are_rejected(tmp_path):
    cache = make_cache()
    cache["clip_probs_train"][0, 0] = .9
    path = tmp_path / "bad.pt"
    torch.save(cache, path)
    with pytest.raises(ValueError, match="Invalid train source probabilities"):
        compare_cache(path, "clip", "toy", tmp_path / "bad-output")
    assert not (tmp_path / "bad-output").exists()
    cache = make_cache()
    cache["train_sample_ids"] = list(reversed(cache["sample_ids"]["train"]))
    with pytest.raises(ValueError, match="aliases disagree"):
        stable_sample_order(cache, 10, "cache-sha")
    fallback, kind = stable_sample_order({}, 2, "cache-sha")
    assert kind == "cache_row_index_only"
    assert fallback == ["cache-sha256:cache-sha/train-row:0", "cache-sha256:cache-sha/train-row:1"]
