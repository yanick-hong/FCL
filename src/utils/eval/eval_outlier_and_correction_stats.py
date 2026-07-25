# -*- coding: utf-8 -*-
"""
eval_outlier_and_correction_stats.py

- 读取 CLIP 特征缓存与 make_obs_labels.py 生成的观测/校正标签文件
- 统计：
  (1) 离群样本（优先使用 outlier_mask/idx_outliers，否则用 s==0）在 CLIP zero-shot 下的预测错误率
  (2) 校正后的整体标签准确率：acc(y_obs vs y_true)
- 兼容性：
  * 优先使用 cache['clip_argmax_train']，否则从 cache['clip_probs_train'].argmax(dim=-1) 回退
  * obs 若含有 'outlier_mask' 或 'idx_outliers' 会被优先采用；否则以 s==0 作为离群样本集合
"""

import os
import argparse
import json
import torch

def load_cache(cache_path: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    print(f"[Load] cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")

    # 类型整理
    def to_i64_if_tensor(x): 
        return x.long() if isinstance(x, torch.Tensor) else x
    def to_f32_if_tensor(x): 
        return x.float() if isinstance(x, torch.Tensor) else x

    for k in ["train_feats","val_feats","test_feats",
              "clip_probs_train","clip_probs_val","clip_probs_test",
              "clip_label_embeds"]:
        if k in cache and isinstance(cache[k], torch.Tensor):
            cache[k] = to_f32_if_tensor(cache[k])

    for k in ["train_labels","val_labels","test_labels",
              "clip_argmax_train","clip_argmax_val","clip_argmax_test",
              "train_idx_from_train50k","val_idx_from_train50k"]:
        if k in cache and isinstance(cache[k], torch.Tensor):
            cache[k] = to_i64_if_tensor(cache[k])

    # 取 GT & zero-shot 预测
    if "train_labels" not in cache:
        raise KeyError("Cache 缺少 'train_labels'")
    y_true = cache["train_labels"].long()

    if "clip_argmax_train" in cache and isinstance(cache["clip_argmax_train"], torch.Tensor):
        clip_pred = cache["clip_argmax_train"].long()
    elif "clip_probs_train" in cache and isinstance(cache["clip_probs_train"], torch.Tensor):
        clip_pred = cache["clip_probs_train"].argmax(dim=-1).long()
    else:
        raise KeyError("Cache 未找到 zero-shot 预测：既无 'clip_argmax_train' 也无 'clip_probs_train'")

    return y_true, clip_pred

def load_obs(obs_path: str, N_expected: int):
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Obs labels not found: {obs_path}")
    print(f"[Load] obs: {obs_path}")
    obs = torch.load(obs_path, map_location="cpu")

    if "y_obs" not in obs or "s" not in obs:
        raise KeyError(f"Obs 需包含 'y_obs' 和 's'，实际 keys={list(obs.keys())}")

    y_obs = obs["y_obs"].long()
    s     = obs["s"].long()

    if y_obs.numel() != N_expected or s.numel() != N_expected:
        raise RuntimeError(f"长度不匹配：N={N_expected}, y_obs={y_obs.numel()}, s={s.numel()}")

    # 离群集合优先级：outlier_mask -> idx_outliers -> s==0
    if "outlier_mask" in obs and isinstance(obs["outlier_mask"], torch.Tensor):
        outlier_mask = obs["outlier_mask"].bool()
        if outlier_mask.numel() != N_expected:
            raise RuntimeError(f"outlier_mask 长度不匹配：{outlier_mask.numel()} vs {N_expected}")
    elif "idx_outliers" in obs:
        idx = obs["idx_outliers"]
        if isinstance(idx, torch.Tensor):
            idx = idx.view(-1).long().tolist()
        outlier_mask = torch.zeros(N_expected, dtype=torch.bool)
        for i in idx:
            if 0 <= int(i) < N_expected: outlier_mask[int(i)] = True
    else:
        # 默认：s==0 为离群&已校正样本
        outlier_mask = (s == 0)

    return y_obs, s, outlier_mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_path", type=str, required=True,
                    help="CLIP 特征缓存 .pt（包含 train_labels 与 clip_*_train）")
    ap.add_argument("--obs_labels_path", type=str, required=True,
                    help="make_obs_labels.py 生成的 .pt（至少含 y_obs, s）")
    ap.add_argument("--save_json", type=str, default="",
                    help="可选：将统计结果保存为 JSON 路径")
    args = ap.parse_args()

    # 1) 读取缓存（GT 与 zero-shot 预测）
    y_true, clip_pred = load_cache(args.cache_path)
    N = y_true.numel()
    print(f"[Info] N_train={N} | classes≈{int(y_true.max().item()+1)}")

    # 2) 读取 obs / 离群集合
    y_obs, s, outlier_mask = load_obs(args.obs_labels_path, N_expected=N)

    num_outliers = int(outlier_mask.sum().item())
    num_s0 = int((s==0).sum().item())
    num_s1 = int((s==1).sum().item())
    print(f"[Split] outliers={num_outliers} | s==0={num_s0} | s==1={num_s1}")

    # 3) 统计
    # 3.1 离群样本（被筛选出的集合）在 zero-shot 下的预测错误率
    if num_outliers > 0:
        err_outliers = (clip_pred[outlier_mask] != y_true[outlier_mask]).float().mean().item()
        acc_outliers = 1.0 - err_outliers
    else:
        err_outliers = float("nan")
        acc_outliers = float("nan")

    # （可选）补充非离群样本错误率，便于对比
    non_out_mask = ~outlier_mask
    if non_out_mask.any():
        err_non_out = (clip_pred[non_out_mask] != y_true[non_out_mask]).float().mean().item()
    else:
        err_non_out = float("nan")

    # 3.2 校正后的整体标签准确率（y_obs vs y_true）
    acc_after = (y_obs == y_true).float().mean().item()

    # （可选）s 分组准确率
    acc_s0 = (y_obs[s==0] == y_true[s==0]).float().mean().item() if (s==0).any() else float("nan")
    acc_s1 = (y_obs[s==1] == y_true[s==1]).float().mean().item() if (s==1).any() else float("nan")

    # 4) 打印
    def pct(x): 
        return "nan" if (x!=x) else f"{x*100:.2f}%"

    print("\n=== Results ===")
    print(f"[Zero-shot @ Outliers]  预测错误率 = {pct(err_outliers)} | 准确率 = {pct(acc_outliers)}")
    print(f"[Zero-shot @ Non-Out]   预测错误率 = {pct(err_non_out)}")
    print(f"[Label Acc After Fix]   全体校正后标签准确率 = {pct(acc_after)}")
    print(f"[Label Acc Grouped]     s==0: {pct(acc_s0)} | s==1: {pct(acc_s1)}")

    # 5) 保存 JSON（可选）
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        out = {
            "N_train": N,
            "num_outliers": num_outliers,
            "num_s0": num_s0,
            "num_s1": num_s1,
            "zero_shot_err_rate_outliers": None if err_outliers!=err_outliers else err_outliers,  # nan 处理
            "zero_shot_err_rate_non_outliers": None if err_non_out!=err_non_out else err_non_out,
            "label_acc_after_correction": acc_after,
            "label_acc_s0": None if acc_s0!=acc_s0 else acc_s0,
            "label_acc_s1": None if acc_s1!=acc_s1 else acc_s1,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[Save] JSON -> {args.save_json}")

if __name__ == "__main__":
    main()
