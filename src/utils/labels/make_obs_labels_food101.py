# -*- coding: utf-8 -*-
"""
make_obs_labels_food101.py

- 为 Food-101 训练集生成观测标签与可信标记（y_obs, s），与现有流水线完全兼容
- 思路：
  1) 以 CLIP zero-shot 在训练集上的预测 y_pseudo（来自 clip_probs_train.argmax 或 clip_argmax_train）作为初始标签；
  2) 逐类（按 y_pseudo 分组）计算特征中心并得到每个样本的“到中心的余弦距离”；
  3) 在每个预测类中，选出距离最大的 top_outlier_pct 样本作为“需人工校正”的样本 —— 将其 y_obs 设为 y_true，s=0（可信）；
     其余样本 y_obs 仍为 y_pseudo，s=1（不可信）。
- 生成的 .pt 文件仅含 {'y_obs': LongTensor(N), 's': LongTensor(N)}，正好被训练脚本的 load_obs_labels() 读取。

使用示例：
python src/utils/labels/make_obs_labels_food101.py \
  --cache_path "${DATA_ROOT}/processed/food101_clip_vit-l-14_embeddings.pt" \
  --save_path  "${DATA_ROOT}/processed/food101_observed_labels.pt" \
  --top_outlier_pct 0.3 \
  --dump_csv "${DATA_ROOT}/processed/food101_obs_preview.csv"

说明：
- 本脚本仅作用于缓存中的 train split（与训练脚本一致）。
- s 的约定：0 表示可信（会用 CE 与 y_true 训练），1 表示不可信（参与 AUC 部分）。
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


def load_clip_cache(cache_path: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

    # 类型与键的健壮性处理（与训练脚本习惯保持一致）
    def _to_fp32(x): return x.to(torch.float32)
    def _to_i64(x): return x.to(torch.long)

    needed_fp32 = ["train_feats", "clip_probs_train"]
    alt_probs_key = "clip_argmax_train"  # 若没有 probs，可退化使用 argmax
    needed_i64 = ["train_labels"]

    for k in needed_fp32:
        if k not in data:
            if k == "clip_probs_train" and alt_probs_key in data:
                pass  # 可接受
            else:
                raise KeyError(f"Cache missing key: '{k}'")
    for k in needed_i64:
        if k not in data:
            raise KeyError(f"Cache missing key: '{k}'")

    feats = _to_fp32(data["train_feats"])
    y_true = _to_i64(data["train_labels"])

    if "clip_probs_train" in data:
        y_pseudo = _to_fp32(data["clip_probs_train"]).argmax(dim=-1).to(torch.long)
    else:
        y_pseudo = _to_i64(data["clip_argmax_train"])

    # 统计类别数
    if "clip_label_embeds" in data:
        C = int(data["clip_label_embeds"].size(0))
    else:
        C = int(max(int(y_true.max()), int(y_pseudo.max())) + 1)

    return feats, y_true, y_pseudo, C


def build_obs_labels_by_outliers(
    feats: torch.Tensor, y_true: torch.Tensor, y_pseudo: torch.Tensor,
    num_classes: int, top_pct: float
):
    """
    在每个预测类（按 y_pseudo 分组）中，按“到类中心的(1-cosine)”距离从大到小选出 top_pct 的样本进行“人工校正”。
    返回：y_obs(LongTensor), s(LongTensor)
    规则：
      - 选中的样本：y_obs = y_true, s=0（可信）
      - 其他样本：  y_obs = y_pseudo, s=1（不可信）
    """
    N, D = feats.size(0), feats.size(1)
    assert y_true.numel() == N and y_pseudo.numel() == N

    feats = F.normalize(feats.to(torch.float32), dim=-1)  # 确保单位化
    y_true = y_true.to(torch.long)
    y_pseudo = y_pseudo.to(torch.long)

    # 初始化
    y_obs = y_pseudo.clone()
    s = torch.ones(N, dtype=torch.long)  # 默认不可信（s=1）

    for c in range(num_classes):
        idx = torch.nonzero(y_pseudo == c, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        F_c = feats[idx]  # (Nc, D)
        center = F.normalize(F_c.mean(dim=0, keepdim=True), dim=-1)  # (1, D)

        # 余弦距离：1 - cos_sim
        cos_sim = (F_c @ center.t()).squeeze(1)  # (Nc,)
        dist = 1.0 - cos_sim

        k = max(1, int(math.ceil(len(idx) * float(top_pct)))) if top_pct > 0 else 0
        if k <= 0:
            continue

        vals, local_top = torch.topk(dist, k, largest=True, sorted=True)
        sel = idx[local_top]  # 映射回全局索引
        # “人工校正”：把这些样本的观测标签改为真值，并标记为可信
        y_obs[sel] = y_true[sel]
        s[sel] = 0

    return y_obs, s


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_path", type=str, default=str(clip_cache("food101")))
    p.add_argument("--save_path",  type=str, default=str(observed_labels_cache("food101")))
    p.add_argument("--top_outlier_pct", type=float, default=0.30, help="每个预测类内标注为“人工校正”的比例（0~1）")
    p.add_argument("--dump_csv", type=str, default="", help="可选：导出预览 CSV（index,y_true,y_pseudo,y_obs,s）")
    return p.parse_args()


def main():
    args = parse_args()

    feats, y_true, y_pseudo, C = load_clip_cache(args.cache_path)
    print(f"[LOAD] train_feats={tuple(feats.shape)} | C={C}")
    print("[INFO] 构造观测标签：按 y_pseudo 分组，挑选类内 top 距离样本做人工校正 -> y_obs=y_true, s=0")

    y_obs, s = build_obs_labels_by_outliers(
        feats=feats, y_true=y_true, y_pseudo=y_pseudo,
        num_classes=C, top_pct=args.top_outlier_pct
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({"y_obs": y_obs.cpu().long(), "s": s.cpu().long()}, args.save_path)

    if args.dump_csv:
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "y_true", "y_pseudo", "y_obs", "s"])
            for i in range(len(y_true)):
                w.writerow([i, int(y_true[i]), int(y_pseudo[i]), int(y_obs[i]), int(s[i])])

    # 汇总
    num_trusted = int((s == 0).sum().item())
    N = int(s.numel())
    print(f"[DONE] Saved obs to: {args.save_path}")
    print(f"       Trusted s=0: {num_trusted} | Untrusted s=1: {N - num_trusted} | top_outlier_pct={args.top_outlier_pct}")


if __name__ == "__main__":
    main()
