# -*- coding: utf-8 -*-
"""
make_obs_labels_caltech101.py
用于 Caltech-101 resnet.pt → 生成 y_obs / s
"""

import os
import math
import argparse
import torch
import torch.nn.functional as F


def load_cache(cache_path: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

    data["train_feats"] = data["train_feats"].float()
    data["train_labels"] = data["train_labels"].long()

    if "clip_argmax_train" in data:
        data["clip_argmax_train"] = data["clip_argmax_train"].long()
    else:
        data["clip_argmax_train"] = data["clip_probs_train"].argmax(dim=1)

    return data


@torch.no_grad()
def outlier_correction(feats, y_true, y_pseudo, top_pct=0.3):
    N, D = feats.shape
    feats = F.normalize(feats, dim=-1)

    C = int(y_true.max().item()) + 1

    y_obs = y_pseudo.clone()
    s = torch.ones(N, dtype=torch.long)

    for c in range(C):
        idx = (y_pseudo == c).nonzero(as_tuple=False).flatten()
        if len(idx) == 0:
            continue
        f = feats[idx]
        center = F.normalize(f.mean(0, keepdim=True), dim=-1)
        dist = 1 - (f @ center.t()).squeeze()

        k = max(1, int(math.ceil(len(idx) * top_pct)))
        _, top_idx = torch.topk(dist, k)
        gidx = idx[top_idx]

        y_obs[gidx] = y_true[gidx]
        s[gidx] = 0

    return y_obs, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="./cache/caltech101/resnet.pt")
    parser.add_argument("--save_path", default="./cache/caltech101/resnet_caltech101_obs_labels.pt")
    parser.add_argument("--top_outlier_pct", type=float, default=0.30)
    args = parser.parse_args()

    cache = load_cache(args.cache_path)

    feats = cache["train_feats"]
    y_true = cache["train_labels"]
    y_pseudo = cache["clip_argmax_train"]

    y_obs, s = outlier_correction(feats, y_true, y_pseudo, args.top_outlier_pct)

    torch.save({
        "y_obs": y_obs,
        "s": s,
        "meta": {
            "N": int(len(s)),
            "pct": args.top_outlier_pct,
            "from_cache": args.cache_path
        }
    }, args.save_path)

    print(f"[DONE] Saved obs_path to: {args.save_path}")
    print(f"       Trusted s=0: {(s==0).sum().item()} | Untrusted: {(s==1).sum().item()} | N={len(s)}")


if __name__ == "__main__":
    main()
