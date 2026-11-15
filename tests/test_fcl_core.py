"""CPU checks for the objective, data contract, sampler and training lifecycle."""
import copy
import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fcl import train_auc_ce as training
from fcl.auc_loss import soft_pairwise_auc


def reference_auc(logits, prior, s, mu, tau, eps=1e-12):
    """Literal two-orientation formula, independent of the optimized backward."""
    scores = logits[s == 1]
    q = mu * scores.softmax(-1).detach() + (1 - mu) * prior[s == 1].detach()
    numerator = logits.sum() * 0
    denominator = logits.new_zeros(())
    mask = ~torch.eye(len(scores), dtype=torch.bool)
    for c in range(logits.shape[1]):
        diff = scores[:, c, None] - scores[None, :, c]
        forward = q[:, c, None] * (1 - q[None, :, c])
        reverse = (1 - q[:, c, None]) * q[None, :, c]
        numerator = numerator + ((forward * F.softplus(-diff / tau)
                                  + reverse * F.softplus(diff / tau)) * mask).sum()
        denominator = denominator + ((forward + reverse) * mask).sum()
    return numerator / (denominator + eps)


def small_cache():
    generator = torch.Generator().manual_seed(7)
    bundle = {}
    for split, count in (("train", 24), ("val", 9), ("test", 9)):
        labels = torch.arange(count) % 3
        features = F.one_hot(labels, 3).float() + .05 * torch.randn(count, 3, generator=generator)
        bundle[f"{split}_feats"] = features
        bundle[f"{split}_labels"] = labels
        bundle[f"clip_probs_{split}"] = (3 * features).softmax(-1)
    return bundle


class FCLCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_chunked_auc_value_gradient_and_detached_posterior(self):
        for mu in (0., .5, 1.):
            for tau in (.3, 1.2):
                torch.manual_seed(42)
                scores = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
                prior = torch.randn(11, 7, dtype=torch.float64).softmax(-1).requires_grad_()
                flags = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
                expected = reference_auc(scores, prior, flags, mu, tau, eps=1e-6)
                expected_gradient = torch.autograd.grad(expected, scores)[0]
                actual = soft_pairwise_auc(scores, prior, flags, mu, tau, chunk_size=3, eps=1e-6)
                gradient, prior_gradient = torch.autograd.grad(actual, (scores, prior), allow_unused=True)
                torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
                torch.testing.assert_close(gradient, expected_gradient, atol=1e-11, rtol=1e-11)
                self.assertIsNone(prior_gradient)
                self.assertEqual(float(gradient[flags == 0].abs().sum()), 0.)
                compatible = training.auc_noise_only_a11_loss(scores, prior, flags, mu, tau, 1e-6)
                torch.testing.assert_close(compatible, expected, atol=1e-11, rtol=1e-11)

    def test_no_pair_objective_has_connected_zero_gradient(self):
        for flags in (torch.zeros(3, dtype=torch.long), torch.tensor([0, 1, 0])):
            logits = torch.randn(3, 2, requires_grad=True)
            result = training.auc_noise_only_a11_loss(logits, torch.full_like(logits, .5), flags)
            self.assertEqual(result.item(), 0.)
            self.assertEqual(torch.autograd.grad(result, logits)[0].abs().sum().item(), 0.)

    def test_sampler_epochs_replay_and_historical_auto_advance(self):
        labels = torch.arange(120) % 6
        first = training.PKBatchSampler(labels, 3, 4, seed=19)
        twin = training.PKBatchSampler(labels, 3, 4, seed=19)
        epoch0, epoch1 = list(first), list(first)
        self.assertEqual(epoch0, list(twin))
        self.assertEqual(epoch1, list(twin))
        class_order = lambda batches: [[int(labels[b[i]]) for i in range(0, 12, 4)] for b in batches]
        self.assertNotEqual(class_order(epoch0), class_order(epoch1))
        first.set_epoch(0)
        self.assertEqual(epoch0, list(first))
        for batch in epoch1:
            counts = labels[batch].bincount()
            self.assertEqual(len(batch), 12)
            self.assertEqual(sorted(counts[counts > 0].tolist()), [4, 4, 4])

    def test_cache_and_observation_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.pt"
            original = small_cache()
            torch.save(original, cache_path)
            self.assertEqual(training.load_clip_cache_strict(str(cache_path))["train_feats"].shape, (24, 3))
            invalid = []
            for key, value in (
                ("val_feats", torch.randn(9, 4)),
                ("train_labels", torch.full((24,), 3)),
                ("test_labels", torch.full((9,), .5)),
                ("clip_probs_train", torch.ones(24, 3)),
                ("clip_probs_test", torch.full((9, 3), float("nan"))),
                ("train_feats", torch.full((24, 3), float("inf"))),
            ):
                bad = copy.deepcopy(original)
                bad[key] = value
                invalid.append(bad)
            for bad in invalid:
                torch.save(bad, cache_path)
                with self.assertRaises(ValueError):
                    training.load_clip_cache_strict(str(cache_path))
            obs_path = Path(directory) / "obs.pt"
            for y_obs, s in (
                (torch.zeros(24, 1), torch.zeros(24)),
                (torch.zeros(24), torch.full((24,), 2)),
                (torch.full((24,), .5), torch.zeros(24)),
                (torch.full((24,), 3), torch.zeros(24)),
                (torch.zeros(23), torch.zeros(24)),
            ):
                torch.save({"y_obs": y_obs, "s": s}, obs_path)
                with self.assertRaises(ValueError):
                    training.load_obs_labels(str(obs_path), expected_size=24, num_classes=3)

    def test_zero_patience_keeps_fixed_epoch_training_and_best_checkpoint(self):
        stopper = training.EarlyStopping(patience=0)
        self.assertFalse(any(stopper.step(1., .5) for _ in range(100)))
        self.assertEqual(stopper.best, .5)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            cache = small_cache()
            torch.save(cache, destination / "cache.pt")
            torch.save({"y_obs": cache["train_labels"], "s": torch.arange(24) % 2}, destination / "obs.pt")
            with patch.object(training, "device", "cpu"):
                head, summary = training.train(
                    str(destination / "cache.pt"), str(destination / "obs.pt"),
                    P=3, K=4, lr=.01, lr_max=.03, weight_decay=1e-4,
                    max_epochs=3, lam=1., mu=.5, seed=42, print_every=0,
                    best_ckpt_path=str(destination / "best.ckpt"), patience=0,
                    min_delta=0., val_dir=str(destination / "logs"), tau=.7, num_workers=0,
                )
                loss, accuracy = training.evaluate(head, cache["test_feats"], cache["test_labels"], num_workers=0)
            self.assertEqual(summary["epochs_completed"], 3)
            self.assertTrue(math.isfinite(loss))
            self.assertTrue(0 <= accuracy <= 1)
            self.assertTrue((destination / "last.ckpt").exists())
            saved = torch.load(destination / "best.ckpt", weights_only=True)
            for key, value in head.state_dict().items():
                torch.testing.assert_close(value, saved[key])
            with (destination / "logs" / "val_metrics.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)


if __name__ == "__main__":
    unittest.main()
