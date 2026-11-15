"""Regression coverage for observed-label supervision and selected checkpoints."""
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.contrast import train_dcd, train_nlprompt


def test_dcd_initial_centers_use_observed_labels():
    # The first point's hidden ground truth could be class 0, but a deliberately
    # conflicting observed label must control its center assignment.
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    observed = torch.tensor([1, 0, 1])
    centers = train_dcd.initial_centers(features, observed, classes=2)
    expected = torch.stack((features[observed == 0].mean(0),
                            features[observed == 1].mean(0)))
    torch.testing.assert_close(centers, F.normalize(expected, dim=1))


def test_nlprompt_tests_best_validation_checkpoint(monkeypatch, tmp_path: Path):
    # Force a better first epoch followed by a worse second one.  The test
    # evaluator must see the first epoch's parameters even though last.ckpt
    # preserves the second epoch for audit.
    bundle = {
        "train_feats": F.normalize(torch.tensor([[1.0, 0.2], [0.1, 1.0]]), dim=1),
        "val_feats": torch.eye(2), "test_feats": torch.eye(2),
        "train_labels": torch.tensor([1, 0]),
        "val_labels": torch.tensor([0, 1]), "test_labels": torch.tensor([0, 1]),
        "y_obs": torch.tensor([0, 1]), "s": torch.tensor([0, 0]),
        "num_classes": 2, "clip_label_embeds": torch.eye(2),
    }
    args = SimpleNamespace(dataset="fixture", cache="cache.pt", obs_labels="obs.pt",
                           run_dir=str(tmp_path), epochs=2, batch_size=2, lr=.05,
                           weight_decay=0., mae_weight=1., patience=3, seed=3)
    monkeypatch.setattr(train_nlprompt, "parse_args", lambda: args)
    monkeypatch.setattr(train_nlprompt, "load_cache_and_observed", lambda *_: bundle)
    monkeypatch.setattr(train_nlprompt, "make_run_dir", lambda *_: tmp_path)
    # Keep the fixture synchronous and independent of multiprocessing start mode.
    original_loader = train_nlprompt.DataLoader
    monkeypatch.setattr(train_nlprompt, "DataLoader", lambda dataset, **kwargs:
                        original_loader(dataset, **{**kwargs, "num_workers": 0,
                                                    "pin_memory": False}))
    for name in ("save_history", "save_config", "save_result", "save_run_metadata"):
        monkeypatch.setattr(train_nlprompt, name, lambda *_, **__: None)

    calls = []

    def evaluate(model, *_):
        snapshot = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        calls.append(snapshot)
        return {"accuracy": .9 if len(calls) == 1 else .1, "nll": 1.}

    monkeypatch.setattr(train_nlprompt, "evaluate", evaluate)
    train_nlprompt.main()
    assert len(calls) == 3  # validation, validation, test
    for key in calls[0]:
        torch.testing.assert_close(calls[-1][key], calls[0][key])
    assert any(not torch.equal(calls[0][key], calls[1][key]) for key in calls[0])
    last = torch.load(tmp_path / "last.ckpt", weights_only=True)["state_dict"]
    for key in last:
        torch.testing.assert_close(last[key].cpu(), calls[1][key])
