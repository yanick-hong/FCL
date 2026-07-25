# -*- coding: utf-8 -*-
"""
make_obs_labels_dtd.py

- 输入：由 extract_CLIP_dtd.py 生成的缓存 .pt
- 输出：obs_path .pt，包含：
    "y_obs": (N,) LongTensor
    "s":     (N,) LongTensor   # 0=可信(已校正), 1=不可信(未校正)
    "meta":  记录来源与比例等信息
- 流程：按 y_pseudo（CLIP 零样本 argmax）分组，计算组内类中心（L2），以 1-cos 为距离，
       选取每组 top_outlier_pct 的样本作为离群点，用 y_true 覆盖并置 s=0。
"""

import os
import math
import argparse
import csv
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import clip_cache, observed_labels_cache

# ----------------------- 读取缓存并统一 dtypes -----------------------
def load_clip_cache(cache_path: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

    # 与训练脚本保持一致的键与类型（存在才转换）
    fp32_keys = [
        "train_feats", "val_feats", "test_feats",
        "clip_probs_train", "clip_probs_val", "clip_probs_test",
        "clip_label_embeds",
    ]
    long_keys = [
        "train_labels", "val_labels", "test_labels",
        "clip_argmax_train", "clip_argmax_val", "clip_argmax_test",
        "train_idx_from_train50k", "val_idx_from_train50k",
    ]
    for k in fp32_keys:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(torch.float32)
    for k in long_keys:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].long()
    return data

# ----------------------- 人工校正（类中心离群） -----------------------
@torch.no_grad()
def outlier_correction_by_class_centers(
    feats: torch.Tensor,    # (N,D) 训练集特征
    y_true: torch.Tensor,   # (N,)   训练集真值标签
    y_pseudo: torch.Tensor, # (N,)   CLIP 零样本 argmax
    top_pct: float = 0.30
):
    """
    1) 特征 L2 归一化
    2) 初始化 y_obs = y_pseudo, s=1(不可信)
    3) 以 y_pseudo 分组：求组内类中心(单位化)，距离=1-cos_sim
    4) 每组选前 top_pct 作为离群点：用 y_true 覆盖，记 s=0(可信)
    """
    N, D = feats.shape
    C = int(y_true.max().item()) + 1
    feats = F.normalize(feats, dim=-1)

    y_obs = y_pseudo.clone()
    s = torch.ones(N, dtype=torch.long)  # 默认不可信（s=1）

    for c in range(C):
        idx = torch.nonzero(y_pseudo == c, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        F_c = feats[idx]                                  # (Nc,D)
        center = F.normalize(F_c.mean(dim=0, keepdim=True), dim=-1)  # (1,D)
        cos_sim = (F_c @ center.t()).squeeze(1)           # (Nc,)
        dist = 1.0 - cos_sim

        k = max(1, int(math.ceil(len(idx) * top_pct)))
        _, topk_ind = torch.topk(dist, k, largest=True, sorted=False)
        outlier_idx_global = idx[topk_ind]

        # 人工校正：用真值替换并记为可信
        y_obs[outlier_idx_global] = y_true[outlier_idx_global]
        s[outlier_idx_global] = 0

    return y_obs, s

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser("Generate observed labels (obs_path) for DTD with manual correction")
    p.add_argument("--cache_path", type=str, default=str(clip_cache("dtd")))
    p.add_argument("--save_path", type=str, default=str(observed_labels_cache("dtd")))
    p.add_argument("--top_outlier_pct", type=float, default=0.30)
    p.add_argument("--dump_csv", type=str, default=None,
                   help="可选：导出检查用 CSV（index,y_true,y_pseudo,y_obs,s）")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # 读取缓存
    cache = load_clip_cache(args.cache_path)

    # 取训练集特征/标签与 CLIP 伪标签
    train_feats = cache["train_feats"]            # (N,D)
    y_true = cache["train_labels"]                # (N,)
    if "clip_argmax_train" in cache and cache["clip_argmax_train"] is not None:
        y_pseudo = cache["clip_argmax_train"]     # (N,)
    else:
        y_pseudo = cache["clip_probs_train"].argmax(dim=1)  # 兜底

    # 生成 y_obs 与 s
    y_obs, s = outlier_correction_by_class_centers(
        feats=train_feats, y_true=y_true, y_pseudo=y_pseudo,
        top_pct=args.top_outlier_pct
    )

    # 保存 obs_path（训练脚本只需 y_obs 与 s）
    torch.save(
        {
            "y_obs": y_obs.cpu().long(),
            "s": s.cpu().long(),
            "meta": {
                "dataset": "DTD",
                "top_outlier_pct": float(args.top_outlier_pct),
                "N": int(y_obs.numel()),
                "from_cache": os.path.abspath(args.cache_path),
            },
        },
        args.save_path,
    )

    # 可选导出检查 CSV
    if args.dump_csv:
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "y_true", "y_pseudo", "y_obs", "s"])
            for i in range(len(y_true)):
                w.writerow([i, int(y_true[i]), int(y_pseudo[i]), int(y_obs[i]), int(s[i])])

    # 简报
    num_trusted = int((s == 0).sum().item())
    N = int(s.numel())
    print(f"[DONE] Saved obs_path to: {args.save_path}")
    print(f"       Trusted s=0: {num_trusted}  |  Untrusted s=1: {N - num_trusted}  |  top_outlier_pct={args.top_outlier_pct}")

if __name__ == "__main__":
    main()
