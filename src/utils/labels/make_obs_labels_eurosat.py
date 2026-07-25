# -*- coding: utf-8 -*-
"""
make_obs_labels_eurosat.py

- 读取由 extract_CLIP_eurosat.py 生成的缓存（.pt）
- 自动完成“可疑样本筛选 + 人工校正（用真值替代）+ 生成 obs_path(.pt)”全流程
- 输出格式与 make_obs_labels.py 一致：只保存 {"y_obs": LongTensor, "s": LongTensor}

规则：
1) 基于 CLIP 伪标签 y_pseudo 分组，在每组内按 “1 - cos(样本特征, 该组类中心)” 从大到小取前 top_outlier_pct。
2) 对入选样本执行“人工校正”：y_obs<-y_true 且 s<-0（可信）；其他样本 y_obs<-y_pseudo 且 s<-1（不可信）。

用法示例：
python src/utils/labels/make_obs_labels_eurosat.py \
  --cache_path "${DATA_ROOT}/processed/eurosat_clip_vit-l-14_embeddings.pt" \
  --save_path "${DATA_ROOT}/processed/eurosat_observed_labels.pt" \
  --top_outlier_pct 0.30 \
  --dump_csv "${DATA_ROOT}/processed/eurosat_obs_preview.csv"   # 可选：导出检查用CSV
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


# -----------------------
# 读取缓存并统一 dtypes（与现有 make_obs_labels.py 保持一致）
# -----------------------
def load_clip_cache(cache_path: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

    # 转 dtypes（与训练/标注脚本常用键保持一致）
    fp32_keys = [
        "train_feats", "val_feats", "test_feats",
        "clip_probs_train", "clip_probs_val", "clip_probs_test",
        "clip_label_embeds"
    ]
    long_keys = [
        "train_labels", "val_labels", "test_labels",
        "clip_argmax_train", "clip_argmax_val", "clip_argmax_test",
        "train_idx_from_train50k", "val_idx_from_train50k"
    ]

    for k in fp32_keys:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(torch.float32)
    for k in long_keys:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].long()

    # 关键存在性检查
    for k in ["train_feats", "train_labels"]:
        if k not in data:
            raise KeyError(f"Cache missing key: {k}")
    if ("clip_argmax_train" not in data) and ("clip_probs_train" not in data):
        raise KeyError("Cache missing CLIP predictions: need clip_argmax_train or clip_probs_train")

    return data


# -----------------------
# 依据“类中心余弦距离”自动校正
# -----------------------
def outlier_correction_by_class_centers(
    feats: torch.Tensor,        # (N,D) — 通常已是 L2-normalized
    y_true: torch.Tensor,       # (N,)
    y_pseudo: torch.Tensor,     # (N,)
    top_pct: float = 0.30       # 每个伪标签类内的离群比例
):
    assert feats.ndim == 2 and y_true.ndim == 1 and y_pseudo.ndim == 1
    assert feats.size(0) == y_true.size(0) == y_pseudo.size(0)
    assert 0.0 < top_pct <= 1.0

    N = feats.size(0)
    C = int(max(int(y_true.max().item()), int(y_pseudo.max().item())) + 1)

    # 默认：y_obs = y_pseudo，全部标为不可信 s=1
    y_obs = y_pseudo.clone().long()
    s = torch.ones(N, dtype=torch.long)

    # 逐伪标签类构建“类中心”并找最离群的 top_pct
    for c in range(C):
        idx = torch.nonzero(y_pseudo == c, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        F_c = feats[idx]                                          # (Nc,D)
        center = F.normalize(F_c.mean(dim=0, keepdim=True), dim=-1)  # (1,D)
        cos_sim = (F_c @ center.t()).squeeze(1)                   # (Nc,)
        dist = 1.0 - cos_sim                                      # 越大越离群

        k = max(1, int(math.ceil(len(idx) * top_pct)))
        _, topk_ind = torch.topk(dist, k, largest=True, sorted=False)
        outlier_idx_global = idx[topk_ind]

        # “人工校正”：用真值替换，并标记为可信
        y_obs[outlier_idx_global] = y_true[outlier_idx_global]
        s[outlier_idx_global] = 0

    return y_obs, s


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default=str(clip_cache("EuroSAT")))
    parser.add_argument("--save_path", type=str, default=str(observed_labels_cache("EuroSAT")))
    parser.add_argument("--top_outlier_pct", type=float, default=0.30)
    parser.add_argument("--dump_csv", type=str, default=None,
                        help="（可选）导出检查用 CSV：index,y_true,y_pseudo,y_obs,s")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    cache = load_clip_cache(args.cache_path)

    # 取训练集特征/标签与 CLIP 伪标签
    train_feats = cache["train_feats"]                     # (N,D) — 已经是 float32
    y_true = cache["train_labels"].long()                  # (N,)
    if "clip_argmax_train" in cache and cache["clip_argmax_train"] is not None:
        y_pseudo = cache["clip_argmax_train"].long()
    else:
        y_pseudo = cache["clip_probs_train"].argmax(dim=1).long()

    # 自动“人工校正”
    y_obs, s = outlier_correction_by_class_centers(
        feats=train_feats, y_true=y_true, y_pseudo=y_pseudo,
        top_pct=args.top_outlier_pct
    )

    # 保存（与 make_obs_labels.py 一致：只需 y_obs 与 s）
    torch.save({"y_obs": y_obs, "s": s}, args.save_path)

    # 可选：导出检查用 CSV
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
