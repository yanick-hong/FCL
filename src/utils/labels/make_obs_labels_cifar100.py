# -*- coding: utf-8 -*-
"""Create corrected observed labels from the CIFAR-100 CLIP cache.

The cache contains ground-truth labels for reproducible evaluation and CLIP
pseudo-labels used by the noisy-label training code.  For each pseudo-label
class, the most distant feature vectors are treated as the trusted subset and
their labels are corrected with the dataset labels.  The output format is the
same as the other ``make_obs_labels_*`` scripts: ``y_obs`` and ``s``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import clip_cache, observed_labels_cache


def make_observed_labels(
    features: torch.Tensor,
    true_labels: torch.Tensor,
    pseudo_labels: torch.Tensor,
    top_outlier_pct: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < top_outlier_pct <= 1.0:
        raise ValueError("top_outlier_pct must be in (0, 1]")
    if features.ndim != 2 or true_labels.ndim != 1 or pseudo_labels.ndim != 1:
        raise ValueError("features must be 2-D and labels must be 1-D")
    if not (features.size(0) == true_labels.numel() == pseudo_labels.numel()):
        raise ValueError("feature and label lengths do not match")

    features = F.normalize(features.float(), dim=-1)
    true_labels = true_labels.long()
    pseudo_labels = pseudo_labels.long()
    observed = pseudo_labels.clone()
    trusted = torch.ones_like(observed)

    for class_id in torch.unique(pseudo_labels).tolist():
        indices = torch.where(pseudo_labels == class_id)[0]
        if indices.numel() == 0:
            continue
        center = F.normalize(features[indices].mean(dim=0, keepdim=True), dim=-1)
        distances = 1.0 - (features[indices] @ center.t()).squeeze(1)
        count = max(1, math.ceil(indices.numel() * top_outlier_pct))
        selected = indices[torch.topk(distances, count, largest=True).indices]
        observed[selected] = true_labels[selected]
        trusted[selected] = 0

    return observed, trusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_path", default=str(clip_cache("CIFAR100")))
    parser.add_argument("--save_path", default=str(observed_labels_cache("CIFAR100")))
    parser.add_argument("--top_outlier_pct", type=float, default=0.30)
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"CLIP cache not found: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")
    required = ("train_feats", "train_labels")
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(f"Cache missing keys: {missing}")
    if "clip_argmax_train" in cache:
        pseudo = cache["clip_argmax_train"]
    elif "clip_probs_train" in cache:
        pseudo = cache["clip_probs_train"].argmax(dim=1)
    else:
        raise KeyError("Cache needs clip_argmax_train or clip_probs_train")

    observed, trusted = make_observed_labels(
        cache["train_feats"], cache["train_labels"], pseudo, args.top_outlier_pct
    )
    output = Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "y_obs": observed.cpu().long(),
            "s": trusted.cpu().long(),
            "meta": {
                "dataset": "CIFAR100",
                "top_outlier_pct": float(args.top_outlier_pct),
                "N": int(observed.numel()),
                "from_cache": str(cache_path.resolve()),
            },
        },
        output,
    )
    print(f"[DONE] Saved observed labels to {output}")
    print(f"       Trusted s=0: {(trusted == 0).sum().item()} | Untrusted s=1: {(trusted == 1).sum().item()}")


if __name__ == "__main__":
    main()
