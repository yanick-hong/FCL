"""Checks for the fixed-mask comparison contract and paired training streams."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compare_label_corrections as correction
from scripts import run_controlled_training as controlled


def create_inputs(directory: Path):
    generator = torch.Generator().manual_seed(10)
    cache = {}
    for split, count in (("train", 24), ("val", 9), ("test", 12)):
        truth = torch.arange(count) % 3
        features = F.one_hot(truth, 3).float() + .08 * torch.randn(count, 3, generator=generator)
        source_labels = truth.clone()
        source_labels[::4] = (source_labels[::4] + 1) % 3
        prior = .1 + .7 * F.one_hot(source_labels, 3).float()
        cache[f"{split}_feats"] = features
        cache[f"{split}_labels"] = truth
        cache[f"clip_probs_{split}"] = prior
        cache[f"clip_argmax_{split}"] = source_labels
    cache["sample_ids"] = {split: [f"{split}/{i}" for i in range(len(cache[f"{split}_labels"]))]
                           for split in ("train", "val", "test")}
    cache_path = directory / "cache.pt"
    torch.save(cache, cache_path)
    obs_directory = directory / "fixed_masks"
    correction.compare_cache(cache_path, "clip", "example", obs_directory, .3, (42, 43, 44))
    return cache_path, obs_directory / "prototype.pt"


def arguments(cache_path, obs_path, output, objective="fcl_auc", seed=42):
    return controlled.parse_args(["--cache", str(cache_path), "--obs-labels", str(obs_path),
        "--output-dir", str(output), "--objective", objective, "--seed", str(seed),
        "--epochs", "3", "--P", "3", "--K", "4", "--device", "cpu"])


class ControlledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic)

    def test_loss_uses_corrected_labels_and_soft_prior_on_disjoint_rows(self):
        logits = torch.tensor([[2., 0., -1.], [.1, 1., -.4], [0., -.5, .8]], requires_grad=True)
        observed = torch.tensor([2, 0, 1])
        flags = torch.tensor([0, 1, 1])
        probabilities = torch.tensor([[.2, .3, .5], [.7, .2, .1], [.1, .1, .8]])
        total, trusted, noisy = controlled.batch_objective(logits, observed, flags, probabilities,
                                  objective="ce_soft", lam=.6, mu=.5, tau=.5)
        expected_ce = F.cross_entropy(logits[:1], observed[:1])
        expected_soft = F.cross_entropy(logits[1:], probabilities[1:])
        torch.testing.assert_close(trusted, expected_ce)
        torch.testing.assert_close(noisy, expected_soft)
        torch.testing.assert_close(total, expected_ce + .6 * expected_soft)
        changed_observed = observed.clone()
        changed_observed[1:] = (changed_observed[1:] + 1) % 3
        changed_prior = probabilities.clone()
        changed_prior[0] = torch.tensor([.8, .1, .1])
        for objective in ("ce_soft", "fcl_auc"):
            original = controlled.batch_objective(logits, observed, flags, probabilities,
                                      objective=objective, lam=1., mu=.5, tau=.5)[0]
            changed = controlled.batch_objective(logits, changed_observed, flags, changed_prior,
                                     objective=objective, lam=1., mu=.5, tau=.5)[0]
            torch.testing.assert_close(original, changed)

    def test_fixed_inputs_reject_identity_mask_and_cache_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache_path, obs_path = create_inputs(root)
            original = torch.load(obs_path, weights_only=True)
            args = arguments(cache_path, obs_path, root / "run")
            controlled.load_fixed_inputs(args)
            cases = []
            wrong = copy.deepcopy(original)
            wrong["meta"]["cache_sha256"] = "0" * 64
            cases.append(wrong)
            wrong = copy.deepcopy(original)
            wrong["selected_mask"] = ~wrong["selected_mask"]
            cases.append(wrong)
            wrong = copy.deepcopy(original)
            wrong["train_sample_ids"] = list(reversed(wrong["train_sample_ids"]))
            cases.append(wrong)
            wrong = copy.deepcopy(original)
            index = int(torch.where(wrong["s"].eq(1))[0][0])
            wrong["y_obs"][index] = (wrong["y_obs"][index] + 1) % 3
            cases.append(wrong)
            wrong = copy.deepcopy(original)
            wrong["meta"]["mask_sha256"] = "0" * 64
            cases.append(wrong)
            for invalid in cases:
                torch.save(invalid, obs_path)
                with self.assertRaises(ValueError):
                    controlled.load_fixed_inputs(args)
            torch.save(original, obs_path)

    def test_paired_objectives_share_initialization_sampling_and_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache_path, obs_path = create_inputs(root)
            auc = controlled.run(arguments(cache_path, obs_path, root / "auc", "fcl_auc"))
            ce = controlled.run(arguments(cache_path, obs_path, root / "ce", "ce_soft"))
            other_seed = controlled.run(arguments(cache_path, obs_path, root / "other", "fcl_auc", seed=43))
            for key in ("cache_sha256", "obs_sha256", "mask_sha256", "y_obs_sha256", "sample_order_sha256",
                        "source_sha256", "protocol_sha256", "init_state_sha256", "sampler_sha256"):
                self.assertEqual(auc["fingerprints"][key], ce["fingerprints"][key])
            for key in ("init_state_sha256", "sampler_sha256"):
                self.assertNotEqual(auc["fingerprints"][key], other_seed["fingerprints"][key])
            self.assertEqual(auc["fingerprints"]["protocol_sha256"], other_seed["fingerprints"]["protocol_sha256"])
            for result, path in ((auc, root / "auc"), (ce, root / "ce")):
                self.assertEqual(result["completed_epochs"], 3)
                self.assertEqual(result["status"], "completed")
                self.assertTrue(0 <= result["test_accuracy"] <= 1)
                self.assertTrue((path / "last.ckpt").is_file())
                self.assertTrue((path / "best.ckpt").is_file())
                with (path / "history.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)
                best_row = max(rows, key=lambda row: float(row["val_accuracy"]))
                self.assertEqual(result["best_epoch"], int(best_row["epoch"]))
                self.assertEqual(json.loads((path / "metrics.json").read_text())["test_accuracy"], result["test_accuracy"])
            with self.assertRaises(FileExistsError):
                controlled.run(arguments(cache_path, obs_path, root / "auc"))

    def test_test_evaluation_uses_best_validation_checkpoint_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache_path, obs_path = create_inputs(root)
            returned_states = []
            calls = []
            actual_evaluate = controlled.evaluate
            def instrumented(model, features, labels, device, batch_size):
                calls.append(len(labels))
                if len(labels) == 9:
                    returned_states.append({key: value.clone() for key, value in model.state_dict().items()})
                    accuracy = [.5, .8, .4][len(returned_states)-1]
                    return {"accuracy": accuracy, "nll": 1., "valid_count": 9, "total_count": 9,
                            "macro_f1": accuracy, "ece": .1}
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, returned_states[1][key])
                return actual_evaluate(model, features, labels, device, batch_size)
            with patch.object(controlled, "evaluate", side_effect=instrumented):
                result = controlled.run(arguments(cache_path, obs_path, root / "run"))
            self.assertEqual(calls, [9, 9, 9, 12])
            self.assertEqual(result["best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
