import os
import math
import argparse
import csv
import torch
import torch.nn.functional as F

def load_clip_cache(cache_path: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

    # 转 dtypes（与训练脚本保持一致的键与类型）
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
    return data

@torch.no_grad()
def outlier_correction_by_class_centers(
    feats: torch.Tensor,   # (N,D)
    y_true: torch.Tensor,  # (N,)
    y_pseudo: torch.Tensor,# (N,) —— CLIP argmax
    top_pct: float = 0.30
):
    """
    1) 对特征做单位化
    2) 初始化 y_obs = y_pseudo, s=1(不可信)
    3) 按 y_pseudo 分组：每类求类中心(单位化)，用 1-cos_sim 作为距离
    4) 选取该类距离最大的前 top_pct 作为离群点：用真实标签覆盖，并将 s=0(可信)
    5) 返回 y_obs, s
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
        F_c = feats[idx]  # (Nc,D)
        center = F.normalize(F_c.mean(dim=0, keepdim=True), dim=-1)  # (1,D)
        cos_sim = (F_c @ center.t()).squeeze(1)  # (Nc,)
        dist = 1.0 - cos_sim

        k = max(1, int(math.ceil(len(idx) * top_pct)))
        _, topk_ind = torch.topk(dist, k, largest=True, sorted=False)
        outlier_idx_global = idx[topk_ind]
        # 人工校正：用真值替换并记为可信
        y_obs[outlier_idx_global] = y_true[outlier_idx_global]
        s[outlier_idx_global] = 0
    return y_obs, s

def main():
    parser = argparse.ArgumentParser("Generate observed labels (obs_path) with manual correction")
    parser.add_argument("--cache_path", type=str, default="./cache/CIFAR100/clip_ViTL14.pt",
                        help="路径：extract_CLIP_cifar100.py 导出的 .pt 缓存")
    parser.add_argument("--save_path", type=str, default="./cache/CIFAR100/cifar100_obs_labels.pt",
                        help="输出：obs_path 文件（.pt），内含 y_obs 与 s 两个 LongTensor")
    parser.add_argument("--top_outlier_pct", type=float, default=0.30,
                        help="每个伪标签类别中按距离类中心最远的比例作为离群点进行人工校正")
    parser.add_argument("--dump_csv", type=str, default=None,
                        help="可选：导出检查用 CSV（index,y_true,y_pseudo,y_obs,s）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    cache = load_clip_cache(args.cache_path)

    # 取训练集特征/标签与 CLIP 伪标签
    train_feats = cache["train_feats"]             # (N,D)
    y_true = cache["train_labels"]                 # (N,)
    if "clip_argmax_train" in cache and cache["clip_argmax_train"] is not None:
        y_pseudo = cache["clip_argmax_train"]
    else:
        # 兜底：用概率取 argmax
        y_pseudo = cache["clip_probs_train"].argmax(dim=1)

    # 生成 y_obs 与 s
    y_obs, s = outlier_correction_by_class_centers(
        feats=train_feats, y_true=y_true, y_pseudo=y_pseudo,
        top_pct=args.top_outlier_pct
    )

    # 保存 obs_path（与训练脚本对接：只需 y_obs 与 s）
    torch.save(
        {
            "y_obs": y_obs.cpu().long(),
            "s": s.cpu().long(),
            "meta": {
                "top_outlier_pct": args.top_outlier_pct,
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
