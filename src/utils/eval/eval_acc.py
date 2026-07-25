# -*- coding: utf-8 -*-
"""
eval_linear_head_CIFAR100.py

评估脚本：
- 读取严格 45k/5k/10k CLIP 特征缓存（由 extract_CLIP_cifar100.py 生成）
- 加载线性头 checkpoint（兼容 state_dict / model / 直接权重 等多种格式；自动去掉 module. 前缀；自动识别是否带 bias）
- 在指定划分上评估 CrossEntropy、Top-1、Top-5
- 可选对比 zero-shot CLIP 概率（若缓存中包含 clip_probs_*）

使用示例：
python eval_linear_head_CIFAR100.py   --cache /path/to/cifar100_clip_cache.pt   --ckpt  /path/to/linear_head_ckpt.pt   --split test --batch-size 4096 --compare-clip
"""

import os
import argparse
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import clip_cache, experiment_checkpoint


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Dict[int, float]:
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    accs = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        accs[k] = (correct_k.item() * (100.0 / batch_size))
    return accs


def select_split_tensors(cache: Dict[str, torch.Tensor], split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        feats, labels = cache["train_feats"], cache["train_labels"]
    elif split == "val":
        feats, labels = cache["val_feats"], cache["val_labels"]
    elif split == "test":
        feats, labels = cache["test_feats"], cache["test_labels"]
    else:
        raise ValueError(f"Unknown split: {split}")
    return feats, labels


def zero_shot_from_cache(cache: Dict[str, torch.Tensor], split: str) -> Optional[torch.Tensor]:
    key = f"clip_probs_{split}"
    if key in cache and cache[key] is not None:
        probs = cache[key]
        if probs.dim() == 2:
            row_sum = probs.float().sum(dim=1, keepdim=True)
            if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3):
                probs = F.softmax(probs, dim=1)
        return probs
    return None


def _gather_weight_bias_from_state_dict(state_dict: Dict[str, torch.Tensor],
                                        in_dim: int, num_classes: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    name_priority = ["head", "classifier", "linear", "fc", "proj", "logit", "cls"]
    candidates: List[Tuple[int, str, torch.Tensor]] = []

    for k, v in state_dict.items():
        if not k.endswith(".weight"):
            continue
        if not torch.is_tensor(v) or v.dim() != 2:
            continue

        out_in = tuple(v.shape)
        in_out = (v.shape[1], v.shape[0])

        score = 0
        for i, tag in enumerate(name_priority):
            if tag in k.lower():
                score += (len(name_priority) - i) * 10
        if out_in == (num_classes, in_dim):
            score += 50
        elif out_in == (in_dim, num_classes) or in_out == (num_classes, in_dim):
            score += 40
        if k.count(".") <= 1:
            score += 5

        candidates.append((score, k, v))

    if not candidates:
        raise KeyError("未在 state_dict 中找到形状为 (C,D)/(D,C) 的线性层权重（*.weight）。")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, weight_key, W = candidates[0]

    prefix = weight_key[:-len(".weight")]
    bias_key = prefix + ".bias"
    b = state_dict.get(bias_key, None)
    if b is not None and (not torch.is_tensor(b) or b.dim() != 1):
        b = None

    if W.shape == (num_classes, in_dim):
        pass
    elif W.shape == (in_dim, num_classes):
        W = W.t().contiguous()
    else:
        if W.t().shape == (num_classes, in_dim):
            W = W.t().contiguous()
        else:
            raise RuntimeError(f"找到的权重形状 {tuple(W.shape)} 与 (C,D)/(D,C) 不匹配。")

    return W, b, prefix


def _strip_prefix_from_keys(sd: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        new_sd[new_k] = v
    return new_sd


def load_linear_from_ckpt(ckpt_path: str, in_dim: int, num_classes: int, device: torch.device) -> nn.Linear:
    obj = torch.load(ckpt_path, map_location="cpu")

    # 1) 直接权重
    if torch.is_tensor(obj):
        W = obj
        b = None
        if W.dim() != 2:
            raise RuntimeError(f"直接权重需为二维 Tensor，收到形状 {tuple(W.shape)}")
        if W.shape == (num_classes, in_dim):
            pass
        elif W.shape == (in_dim, num_classes):
            W = W.t().contiguous()
        else:
            raise RuntimeError(f"直接权重形状 {tuple(W.shape)} 与 (C,D)/(D,C) 不匹配。C={num_classes}, D={in_dim}")
        layer = nn.Linear(in_dim, num_classes, bias=False)
        layer.weight.data.copy_(W.float())
        return layer.to(device)

    # 2) state_dict 或 包含 state_dict 的 dict
    if isinstance(obj, dict) and not isinstance(obj, nn.Module):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj

        sd = _strip_prefix_from_keys(sd, ["module.", "model.", "net."])

        W, b, prefix = _gather_weight_bias_from_state_dict(sd, in_dim, num_classes)

        layer = nn.Linear(in_dim, num_classes, bias=(b is not None))
        layer.weight.data.copy_(W.float())
        if b is not None:
            if b.shape[0] != num_classes:
                raise RuntimeError(f"bias 长度 {b.shape[0]} 与类别数 {num_classes} 不一致。")
            layer.bias.data.copy_(b.float())
        return layer.to(device)

    # 3) 完整 nn.Module
    if isinstance(obj, nn.Module):
        mod = obj
        if hasattr(mod, "module") and isinstance(mod.module, nn.Module):
            mod = mod.module

        candidates = []
        for name, m in mod.named_modules():
            if isinstance(m, nn.Linear):
                if m.in_features == in_dim and m.out_features == num_classes:
                    candidates.append((name, m))

        if not candidates:
            sd = mod.state_dict()
            sd = _strip_prefix_from_keys(sd, ["module.", "model.", "net."])
            W, b, _ = _gather_weight_bias_from_state_dict(sd, in_dim, num_classes)
            layer = nn.Linear(in_dim, num_classes, bias=(b is not None))
            layer.weight.data.copy_(W.float())
            if b is not None:
                layer.bias.data.copy_(b.float())
            return layer.to(device)

        prio = ["head", "classifier", "linear", "fc", "logit", "cls"]
        def score(name: str) -> int:
            s = 0
            for i, t in enumerate(prio):
                if t in name.lower():
                    s += (len(prio) - i) * 10
            if name.count(".") <= 1:
                s += 5
            return s

        candidates.sort(key=lambda x: score(x[0]), reverse=True)
        chosen_name, chosen = candidates[0]

        layer = nn.Linear(in_dim, num_classes, bias=(chosen.bias is not None))
        layer.weight.data.copy_(chosen.weight.detach().cpu().float())
        if chosen.bias is not None:
            layer.bias.data.copy_(chosen.bias.detach().cpu().float())
        return layer.to(device)

    raise TypeError("无法识别的 checkpoint 类型：既不是 Tensor、也不是 dict(state_dict)、也不是 nn.Module。")


def evaluate(split: str,
             feats: torch.Tensor,
             labels: torch.Tensor,
             linear: nn.Linear,
             batch_size: int,
             device: torch.device) -> Dict[str, float]:
    ds = TensorDataset(feats, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    ce_loss = 0.0
    n = 0
    top1_total = 0.0
    top5_total = 0.0

    linear.eval()

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            logits = linear(xb)

            loss = F.cross_entropy(logits, yb, reduction="sum")
            accs = compute_topk_accuracy(logits, yb, topk=(1, 5))

            bs = xb.size(0)
            ce_loss += loss.item()
            top1_total += accs[1] * bs / 100.0
            top5_total += accs[5] * bs / 100.0
            n += bs

    return {
        "ce": ce_loss / max(1, n),
        "acc_top1": (top1_total / max(1, n)) * 100.0,
        "acc_top5": (top5_total / max(1, n)) * 100.0,
        "num": n,
    }


def evaluate_probs(split: str,
                   probs: torch.Tensor,
                   labels: torch.Tensor,
                   batch_size: int,
                   device: torch.device) -> Dict[str, float]:
    ds = TensorDataset(probs, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    top1_total = 0.0
    top5_total = 0.0
    n = 0
    with torch.no_grad():
        for pb, yb in dl:
            pb = pb.to(device).float()
            yb = yb.to(device).long()
            logits = torch.log(pb.clamp_min(1e-12))
            accs = compute_topk_accuracy(logits, yb, topk=(1, 5))
            bs = yb.size(0)
            top1_total += accs[1] * bs / 100.0
            top5_total += accs[5] * bs / 100.0
            n += bs

    return {
        "acc_top1": (top1_total / max(1, n)) * 100.0,
        "acc_top5": (top5_total / max(1, n)) * 100.0,
        "num": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate linear head on  CLIP features cache.")
    parser.add_argument("--cache", type=str, default=str(clip_cache("CIFAR100")))
    
    parser.add_argument("--ckpt", type=str, default=str(experiment_checkpoint("CIFAR100_auc_ce")))
    
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare-clip", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    assert os.path.isfile(args.cache), f"Cache not found: {args.cache}"
    cache: Dict[str, torch.Tensor] = torch.load(args.cache, map_location="cpu")

    feats, labels = select_split_tensors(cache, args.split)
    feats = feats.contiguous().float()
    labels = labels.long()

    in_dim = feats.shape[1]
    if "clip_label_embeds" in cache and cache["clip_label_embeds"] is not None:
        num_classes = int(cache["clip_label_embeds"].shape[0])
    elif "class_names" in cache and cache["class_names"] is not None:
        try:
            num_classes = len(cache["class_names"])
        except Exception:
            num_classes = int(max(labels.max().item() + 1, 100))
    else:
        num_classes = int(max(labels.max().item() + 1, 100))

    print(f"[Info] Split={args.split} | feats={tuple(feats.shape)} | classes={num_classes} | device={device}")

    linear = load_linear_from_ckpt(args.ckpt, in_dim=in_dim, num_classes=num_classes, device=device)
    linear.eval()

    metrics = evaluate(args.split, feats, labels, linear, batch_size=args.batch_size, device=device)
    print(f"[Linear] CE={metrics['ce']:.4f} | Top1={metrics['acc_top1']:.2f}% | Top5={metrics['acc_top5']:.2f}% | N={metrics['num']}")

    if args.compare_clip:
        probs = zero_shot_from_cache(cache, args.split)
        if probs is not None:
            zs_metrics = evaluate_probs(args.split, probs, labels, batch_size=args.batch_size, device=device)
            print(f"[ZeroShot-CLIP] Top1={zs_metrics['acc_top1']:.2f}% | Top5={zs_metrics['acc_top5']:.2f}% | N={zs_metrics['num']}")
        else:
            print("[ZeroShot-CLIP] 缓存中未找到 clip_probs_*，跳过对比。")


if __name__ == "__main__":
    main()
