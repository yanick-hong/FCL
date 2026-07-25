# -*- coding: utf-8 -*-
"""
extract_CLIP_caltech101.py

- 自动下载并加载 Caltech-101（排除 BACKGROUND_Google）
- 按类分层划分 Train/Val/Test （优先 30/10/剩余；小类自适应）
- 使用 OpenCLIP ViT-L/14 提取图像特征（D=768），并计算零样本概率
- 输出与 CIFAR100 缓存风格对齐的 .pt 字典，便于后续训练脚本复用：
  keys:
    - train_feats (Ntr,D), train_labels (Ntr,)
    - val_feats, val_labels
    - test_feats, test_labels
    - clip_probs_train/val/test  (N,C)
    - clip_argmax_train/val/test (N,)
    - clip_label_embeds (C,D)
    - class_names (list[str])
    - meta (dict)
"""

import os
import time
import math
import argparse
import random
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision.datasets import Caltech101

import open_clip
from tqdm import tqdm
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import DATA_ROOT, clip_cache


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_prompts(class_names: List[str]) -> List[str]:
    # 你可以按需扩展模板；保持与 CIFAR 一致的简单模板
    return [f"a photo of a {name.replace('_', ' ')}." for name in class_names]


def stratified_split(indices_by_class: Dict[int, List[int]],
                    n_train_per_cls: int = 30,
                    n_val_per_cls: int = 10,
                    seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    优先 30/10/剩余；若类样本不足，退化为 70%/10%/剩余（至少各 1）
    """
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c, idxs in indices_by_class.items():
        idxs = idxs[:]  # copy
        rng.shuffle(idxs)
        n = len(idxs)

        if n >= (n_train_per_cls + n_val_per_cls + 1):
            nt = n_train_per_cls
            nv = n_val_per_cls
        elif n >= 3:
            nt = max(1, int(round(0.7 * n)))
            nv = max(1, int(round(0.1 * n)))
            if nt + nv >= n:  # 保证留出测试
                nv = max(1, min(n - nt - 1, nv))
                if nt + nv >= n:
                    nt = max(1, n - nv - 1)
        else:
            # 极小类兜底：1/1/剩下（可能 1/0/0 但至少可训练）
            nt = max(1, n - 2)
            nv = 1 if n - nt >= 2 else max(0, n - nt - 1)

        tr = idxs[:nt]
        va = idxs[nt:nt + nv]
        te = idxs[nt + nv:]

        train_idx.extend(tr)
        val_idx.extend(va)
        test_idx.extend(te)

    return train_idx, val_idx, test_idx


@torch.no_grad()
def encode_image_set(model, preprocess, dataset, index_list: List[int],
                     text_features: torch.Tensor, device: str,
                     batch_size: int, num_workers: int):
    """
    返回：
      feats: (N,D) 归一化图像特征
      labels: (N,)
      probs: (N,C) 由 CLIP logits softmax
      argmax: (N,)
    """
    sub = Subset(dataset, index_list)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_feats, all_labels, all_probs, all_argmax = [], [], [], []
    logit_scale = model.logit_scale.exp().to(device) if hasattr(model, "logit_scale") else torch.tensor(1.0, device=device)
    text_features = F.normalize(text_features, dim=-1)  # (C,D)

    for images, targets in tqdm(loader, desc="Encoding", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)  # (B,D)

        logits = logit_scale * image_features @ text_features.t()  # (B,C)
        probs = torch.softmax(logits, dim=-1)
        argmax = probs.argmax(dim=-1)

        all_feats.append(image_features.float().cpu())
        all_labels.append(targets.long().cpu())
        all_probs.append(probs.float().cpu())
        all_argmax.append(argmax.long().cpu())

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.cat(all_probs, dim=0)
    argmax = torch.cat(all_argmax, dim=0)
    return feats, labels, probs, argmax


def main():
    parser = argparse.ArgumentParser("Extract CLIP (ViT-L/14) cache for Caltech-101")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--out_path", type=str, default=str(clip_cache("caltech101")))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--model", type=str, default="ViT-L-14")
    parser.add_argument("--force_rebuild", action="store_true", help="忽略已有缓存并重新提取")
    args = parser.parse_args()

    if os.path.exists(args.out_path) and not args.force_rebuild:
        print(f"[CACHE] Reusing existing features: {args.out_path}")
        return
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"[Info] device={device} | model={args.model} | weights={args.pretrained}")

    # 1) 加载 OpenCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # 2) 下载并加载 Caltech-101
    print("[Load] Caltech-101 (may download on first run)...")
    full_ds = Caltech101(root=args.data_root, download=True, transform=preprocess)

    # 3) 过滤 BACKGROUND_Google，并重映射类别 id -> [0..C-1]
    raw_classes: List[str] = list(full_ds.categories) if hasattr(full_ds, "categories") else None
    if raw_classes is None:
        # torchvision 版本兜底（极少数旧版）
        # 构建类别名：从 target_to_class / classes 等属性推断
        if hasattr(full_ds, "classes"):
            raw_classes = list(full_ds.classes)
        else:
            raise RuntimeError("Cannot find class names from torchvision Caltech101; please upgrade torchvision.")

    bg_name = "BACKGROUND_Google"
    keep_names = [c for c in raw_classes if c != bg_name]
    name_to_keep_id = {name: i for i, name in enumerate(keep_names)}

    # 提取 targets（兼容不同 torchvision 版本）
    if hasattr(full_ds, "y"):
        targets_raw = list(full_ds.y)
    elif hasattr(full_ds, "targets"):
        targets_raw = list(full_ds.targets)
    else:
        # 兜底：逐个 __getitem__ 读取（会慢一些，但保证兼容）
        targets_raw = []
        for i in range(len(full_ds)):
            _, y = full_ds[i]
            targets_raw.append(y)

    # 原始 target -> 类名
    idx_to_name = {i: n for i, n in enumerate(raw_classes)}

    # 建立“保留类”的重映射 target 列表与索引列表
    keep_indices = []
    remapped_targets = []
    indices_by_class: Dict[int, List[int]] = {i: [] for i in range(len(keep_names))}
    for idx, t in enumerate(targets_raw):
        name = idx_to_name[int(t)]
        if name == bg_name:
            continue
        new_t = name_to_keep_id[name]
        keep_indices.append(idx)
        remapped_targets.append(new_t)
        indices_by_class[new_t].append(len(remapped_targets) - 1)  # 注意：基于 keep_indices 的相对索引

    # 构建“只含保留样本”的视图（通过 Subset），并让 __getitem__ 返回 remapped_targets
    class RemapTargets(torch.utils.data.Dataset):
        def __init__(self, base: torch.utils.data.Dataset, kept_idx: List[int],
                     kept_targets: List[int]):
            self.base = base
            self.kept_idx = kept_idx
            self.kept_targets = kept_targets
            self.transform = base.transform

        def __len__(self):
            return len(self.kept_idx)

        def __getitem__(self, i):
            img, _ = self.base[self.kept_idx[i]]
            return img, self.kept_targets[i]

    ds = RemapTargets(full_ds, keep_indices, remapped_targets)
    C = len(keep_names)
    print(f"[Info] Caltech-101 (w/o BACKGROUND): classes={C}, samples={len(ds)}")

    # 4) 分层划分
    train_idx, val_idx, test_idx = stratified_split(indices_by_class, n_train_per_cls=30, n_val_per_cls=10, seed=args.seed)
    print(f"[Split] train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

    # 5) 文本编码（类别提示）
    prompts = build_prompts(keep_names)
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features = F.normalize(text_features, dim=-1)  # (C,D)

    # 6) 图像编码
    tr_feats, tr_labels, tr_probs, tr_argmax = encode_image_set(
        model, preprocess, ds, train_idx, text_features, device, args.batch_size, args.num_workers
    )
    va_feats, va_labels, va_probs, va_argmax = encode_image_set(
        model, preprocess, ds, val_idx, text_features, device, args.batch_size, args.num_workers
    )
    te_feats, te_labels, te_probs, te_argmax = encode_image_set(
        model, preprocess, ds, test_idx, text_features, device, args.batch_size, args.num_workers
    )

    # 7) 保存缓存
    out = {
        # features & labels
        "train_feats": tr_feats, "train_labels": tr_labels,
        "val_feats": va_feats,   "val_labels": va_labels,
        "test_feats": te_feats,  "test_labels": te_labels,
        # CLIP zero-shot distributions
        "clip_probs_train": tr_probs, "clip_argmax_train": tr_argmax,
        "clip_probs_val": va_probs,   "clip_argmax_val": va_argmax,
        "clip_probs_test": te_probs,  "clip_argmax_test": te_argmax,
        # class text embeddings
        "clip_label_embeds": text_features.cpu(),
        # meta info
        "class_names": keep_names,
        "prompts": prompts,
        "meta": {
            "dataset": "Caltech-101 (no BACKGROUND_Google)",
            "num_classes": C,
            "model": args.model,
            "pretrained": args.pretrained,
            "feat_dim": int(tr_feats.shape[1]),
            "num_train": int(tr_feats.shape[0]),
            "num_val": int(va_feats.shape[0]),
            "num_test": int(te_feats.shape[0]),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
        },
        # 为了与 CIFAR100 缓存键兼容，这两个键给出 None（Caltech-101 不需要）
        "train_idx_from_train50k": None,
        "val_idx_from_train50k": None,
    }

    torch.save(out, args.out_path)
    print(f"[SAVE] -> {args.out_path}")
    print(f"[Done] D={out['meta']['feat_dim']} | C={C} | train/val/test = "
          f"{out['meta']['num_train']}/{out['meta']['num_val']}/{out['meta']['num_test']}")
    print("      keys:", ", ".join(out.keys()))


if __name__ == "__main__":
    main()
