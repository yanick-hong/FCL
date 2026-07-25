# -*- coding: utf-8 -*-
"""
extract_CLIP_dtd.py  (fixed)

- 从 torchvision.datasets.DTD 下载/载入 DTD（47类）
- 使用 CLIP ViT-L/14 提取 train/val/test 三个 split 的图像特征
- 生成零样本文本标签嵌入，并计算每张图的零样本概率分布
- 输出缓存的键名/数据类型与 extract_CLIP_cifar100.py 对齐
"""

import os, argparse, time
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import RAW_ROOT, clip_cache

try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
except Exception as e:
    raise RuntimeError("未找到 `clip` 包，请先安装：pip install git+https://github.com/openai/CLIP.git") from e

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(RAW_ROOT))
    p.add_argument("--out", type=str, default=str(clip_cache("dtd")))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--download", action="store_true")
    p.add_argument("--force_rebuild", action="store_true", help="忽略已有缓存并重新提取")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backbone", type=str, default="ViT-L/14")
    return p.parse_args()


def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


TEMPLATES = [
    "a photo of a {}.",
]
    # "a close-up photo of a {} texture.",
    # "a photo of the pattern of {}.",
    # "a detailed photo of {}.",
    # "a texture of {}.",
    # "a pattern of {}.",

@torch.no_grad()
def build_label_embeddings(model, tokenizer, classnames: List[str]) -> torch.Tensor:
    zs = []
    for name in classnames:
        texts = [tmpl.format(name.replace("_", " ")) for tmpl in TEMPLATES]
        tok = tokenizer(texts, truncate=True).to(device)
        text_feats = model.encode_text(tok)
        text_feats = F.normalize(text_feats, dim=-1)
        text_feat = F.normalize(text_feats.mean(dim=0, keepdim=True), dim=-1)
        zs.append(text_feat)
    return torch.cat(zs, dim=0).float()  # (C, D)


@torch.no_grad()
def extract_split(model, preprocess, dataset, batch_size: int, num_workers: int, label_embeds: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats_list, labels_list, probs_list, amax_list = [], [], [], []
    logit_scale = model.logit_scale.exp().item()

    for images, labels in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast(False):
            img_feats = model.encode_image(images)
            img_feats = F.normalize(img_feats, dim=-1).float()
        logits = logit_scale * img_feats @ label_embeds.t()
        probs = logits.softmax(dim=-1)
        amax = probs.argmax(dim=-1).long()

        feats_list.append(img_feats.cpu())
        labels_list.append(labels.long().cpu())
        probs_list.append(probs.cpu())
        amax_list.append(amax.cpu())

    feats = torch.cat(feats_list, 0)
    labels = torch.cat(labels_list, 0)
    probs = torch.cat(probs_list, 0)
    amax = torch.cat(amax_list, 0)
    return feats, labels, probs, amax


def main():
    args = parse_args()
    if os.path.exists(args.out) and not args.force_rebuild:
        print(f"[CACHE] Reusing existing features: {args.out}")
        return
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) CLIP
    print(f"[CLIP] Loading model: {args.backbone}")
    model, preprocess = clip.load(args.backbone, device=device)
    model.eval()

    # 2) DTD splits（无 fold 参数）
    print(f"[DATA] Loading DTD from: {args.data_root} | download={args.download}")
    # 某些老版本 torchvision 可能将参数名叫 partition，这里做个兼容
    def load_dtd(split: str):
        try:
            return datasets.DTD(root=args.data_root, split=split, transform=preprocess,
                                download=args.download)
        except TypeError:
            # 兼容：有些版本使用 partition 而不是 split
            return datasets.DTD(root=args.data_root, partition=split, transform=preprocess,
                                download=args.download)

    ds_train = load_dtd("train")
    ds_val   = load_dtd("val")
    ds_test  = load_dtd("test")

    classes = list(ds_train.classes)
    class_to_idx = ds_train.class_to_idx
    assert classes == ds_val.classes == ds_test.classes, "train/val/test 类别顺序不一致"

    C = len(classes)
    print(f"[DATA] Classes: {C}")

    # 3) 文本原型
    print("[CLIP] Building text embeddings ...")
    label_embeds = build_label_embeddings(model, clip.tokenize, classes)  # (C, D)

    # 4) 提取三划分
    print("[EXTRACT] Train split ...")
    train_feats, train_labels, clip_probs_train, clip_argmax_train = extract_split(
        model, preprocess, ds_train, args.batch, args.workers, label_embeds
    )
    print("[EXTRACT] Val split ...")
    val_feats, val_labels, clip_probs_val, clip_argmax_val = extract_split(
        model, preprocess, ds_val, args.batch, args.workers, label_embeds
    )
    print("[EXTRACT] Test split ...")
    test_feats, test_labels, clip_probs_test, clip_argmax_test = extract_split(
        model, preprocess, ds_test, args.batch, args.workers, label_embeds
    )

    # 5) 保存（键名/类型与 CIFAR100 提取脚本一致）
    cache: Dict[str, torch.Tensor] = {
        "train_feats": train_feats.float(),
        "val_feats":   val_feats.float(),
        "test_feats":  test_feats.float(),

        "train_labels": train_labels.long(),
        "val_labels":   val_labels.long(),
        "test_labels":  test_labels.long(),

        "clip_probs_train": clip_probs_train.float(),
        "clip_probs_val":   clip_probs_val.float(),
        "clip_probs_test":  clip_probs_test.float(),

        "clip_argmax_train": clip_argmax_train.long(),
        "clip_argmax_val":   clip_argmax_val.long(),
        "clip_argmax_test":  clip_argmax_test.long(),

        "clip_label_embeds": label_embeds.float(),
    }

    meta = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "dataset": "DTD",
        "backbone": args.backbone,
        "preprocess": "CLIP_default",
        "logit_scale_exp": float(model.logit_scale.exp().item()),
        "seed": int(args.seed),
    }

    torch.save({**cache, "meta": meta}, args.out)
    print(f"[SAVE] Done -> {args.out}")

    def shape(t): return tuple(t.shape)
    print("[SHAPE]")
    print("  train_feats:", shape(cache["train_feats"]), "train_labels:", shape(cache["train_labels"]))
    print("  val_feats:  ", shape(cache["val_feats"]),   "val_labels:  ", shape(cache["val_labels"]))
    print("  test_feats: ", shape(cache["test_feats"]),  "test_labels: ", shape(cache["test_labels"]))
    print("  clip_probs_train:", shape(cache["clip_probs_train"]))
    print("  clip_probs_val:  ", shape(cache["clip_probs_val"]))
    print("  clip_probs_test: ", shape(cache["clip_probs_test"]))
    print("  clip_label_embeds:", shape(cache["clip_label_embeds"]))


if __name__ == "__main__":
    main()
