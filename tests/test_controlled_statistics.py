from copy import deepcopy
import math

import pytest

from scripts.summarize_controlled_results import CONSISTENT, mean_interval, summarize


def runs():
    return [{"status": "completed", "dataset": "toy", "label_source": "clip", "selection": "random",
             "selection_seed": 42, "objective": objective, "seed": seed,
             "test_accuracy": value + (0.01 if objective == "fcl_auc" else 0),
             "metrics_path": f"{objective}/{seed}", "metrics_sha256": "metrics",
             "fingerprints": {**{k: k for k in CONSISTENT}, "init_state_sha256": str(seed), "sampler_sha256": str(seed)}}
            for objective in ("fcl_auc", "ce_soft") for seed, value in zip((42, 43, 44), (.8, .82, .84))]


def test_sample_std_and_student_t_ci_and_paired_difference():
    interval = mean_interval([80, 82, 84])
    assert interval["mean"] == 82
    assert interval["sample_std"] == 2
    assert interval["ci95_high"] - 82 == pytest.approx(4.3026527299 * 2 / math.sqrt(3))
    result = summarize(runs())
    assert len(result["groups"]) == 2
    assert result["paired_objective_differences"][0]["mean"] == pytest.approx(1)


def test_duplicate_missing_or_mixed_input_seeds_rejected():
    examples = [runs() + [deepcopy(runs()[0])], runs()[1:]]
    mixed = runs()
    mixed[0]["fingerprints"]["mask_sha256"] = "other"
    examples.append(mixed)
    for rows in examples:
        with pytest.raises(ValueError):
            summarize(rows)


def test_pairing_requires_same_initialization_and_sampler():
    rows = runs()
    rows[0]["fingerprints"]["sampler_sha256"] = "different"
    with pytest.raises(ValueError, match="sampler_sha256"):
        summarize(rows)
