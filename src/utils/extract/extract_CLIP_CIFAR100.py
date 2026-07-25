# -*- coding: utf-8 -*-
"""
extract_CLIP_cifar100.py

- 用 CLIP (ViT-L/14) 提取 CIFAR-100 特征并缓存
- 从训练集每类严格抽取 50 张做验证 => 45k/5k/10k (train/val/test)
- 保存：特征、标签、CLIP 文本标签嵌入、CLIP 概率与 argmax、以及可复现实验的划分索引

"""

import os
import time
import math
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from clip import clip  # 需可用
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import clip_cache, dataset_root

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# 通用工具
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size=224):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def article(name):
    return "an" if name[0].lower() in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


# -----------------------
# 读取 CIFAR-100 python 版
# -----------------------
def load_cifar100(root: str = str(dataset_root("CIFAR100"))):
    with open(os.path.join(root, "train"), "rb") as f:
        data_train = pickle.load(f, encoding="latin1")
    with open(os.path.join(root, "test"), "rb") as f:
        data_test = pickle.load(f, encoding="latin1")
    with open(os.path.join(root, "meta"), "rb") as f:
        data_meta = pickle.load(f, encoding="latin1")
    return data_train, data_test, data_meta


def read_data_cifar_100(root: str):
    data_train, data_test, data_meta = load_cifar100(root)
    train_data = data_train["data"].reshape((data_train["data"].shape[0], 3, 32, 32))
    test_data = data_test["data"].reshape((data_test["data"].shape[0], 3, 32, 32))
    train_label = np.array(data_train["fine_labels"], dtype=np.int64)
    test_label = np.array(data_test["fine_labels"], dtype=np.int64)
    class_names = data_meta["fine_label_names"]  # 顺序与标签 id 对齐
    return train_data, train_label, test_data, test_label, class_names


# -----------------------
# CLIP 文本标签嵌入
# -----------------------
single_template = ["a photo of a {}."]


@torch.no_grad()
def build_clip_label_embedding(model, categories: List[str]) -> torch.Tensor:
    templates = single_template
    run_on_gpu = torch.cuda.is_available()
    if run_on_gpu:
        model = model.cuda()

    label_embeds = []
    for category in categories:
        texts = [
            template.format(processed_name(category, rm_dot=True), article=article(category))
            for template in templates
        ]
        texts = [
            "This is " + text if text.startswith(("a", "the")) else text
            for text in texts
        ]
        tokens = clip.tokenize(texts)
        if run_on_gpu:
            tokens = tokens.cuda()
        text_embeddings = model.encode_text(tokens)              # (T,D)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        text_embedding = text_embeddings.mean(dim=0)             # (D,)
        text_embedding = F.normalize(text_embedding, dim=0)
        label_embeds.append(text_embedding)
    label_embeds = torch.stack(label_embeds, dim=0)              # (C,D)
    return label_embeds.to(torch.float32)


# -----------------------
# 严格分层验证划分（每类 k=50）
# -----------------------
def stratified_val_split_per_class(labels: np.ndarray, k_per_class: int = 50, seed: int = 42
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    labels = labels.astype(np.int64)
    classes = np.unique(labels)
    val_idx = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        if idx_c.size < k_per_class:
            raise ValueError(f"Class {c} has {idx_c.size} samples < {k_per_class}.")
        sel = np.random.choice(idx_c, size=k_per_class, replace=False)
        val_idx.append(sel)
    val_idx = np.concatenate(val_idx, axis=0)
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[val_idx] = False
    train_idx = np.nonzero(mask)[0]
    # 断言严格均衡
    assert len(val_idx) == k_per_class * len(classes) == 5000
    # 训练 50000 - 5000 = 45000
    assert len(train_idx) == labels.shape[0] - len(val_idx) == 45000
    return train_idx, val_idx


# -----------------------
# Numpy -> Tensor 数据集
# -----------------------
class CIFARArrayDataset(Dataset):
    def __init__(self, imgs: np.ndarray, labels: np.ndarray, tfm):
        self.imgs = imgs
        self.labels = labels.astype(np.int64)
        self.tfm = tfm

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        x = Image.fromarray(np.uint8(self.imgs[idx]).transpose((1, 2, 0)))
        x = self.tfm(x)
        y = int(self.labels[idx])
        return x, y


# -----------------------
# 特征编码
# -----------------------
@torch.no_grad()
def encode_split(clip_model, loader, label_embeds, name="train"):
    feats_list, labels_list, probs_list, argmax_list = [], [], [], []
    total = len(loader.dataset)
    seen = 0
    for xb, yb in loader:
        xb = xb.to(device)
        f = clip_model.encode_image(xb)          # (B,D)
        f = f.to(torch.float32)
        f = F.normalize(f, dim=-1)
        logits_clip = f @ label_embeds.t()       # (B,C)
        probs_clip = F.softmax(logits_clip, dim=-1).to(torch.float32)
        yhat = probs_clip.argmax(dim=-1)

        feats_list.append(f.cpu())
        labels_list.append(yb.clone())
        probs_list.append(probs_clip.cpu())
        argmax_list.append(yhat.cpu())

        seen += xb.size(0)
        print(f"[CLIP-{name}] Encoded {seen}/{total}")

    feats  = torch.cat(feats_list,  dim=0).to(torch.float32)
    labels = torch.cat(labels_list, dim=0).long()
    probs  = torch.cat(probs_list,  dim=0).to(torch.float32)
    argmx  = torch.cat(argmax_list, dim=0).long()
    return feats, labels, probs, argmx


# -----------------------
# 构建或加载缓存（严格 45k/5k/10k）
# -----------------------
@torch.no_grad()
def build_or_load_clip_cache_strict_split(
    root: str = str(dataset_root("CIFAR100")),
    cache_path: str = str(clip_cache("CIFAR100")),
    image_size: int = 224,
    batch_size: int = 256,
    model_name: str = "ViT-L/14",
    seed: int = 42,
    num_workers: int = 2,
    force_rebuild: bool = False,
) -> Dict[str, torch.Tensor]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if (not force_rebuild) and os.path.exists(cache_path):
        try:
            print(f"[CACHE] Loading from: {cache_path}")
            data = torch.load(cache_path, map_location="cpu")
            # 统一 dtypes（向后兼容）
            tensor_fp32_keys = [
                "train_feats", "val_feats", "test_feats",
                "clip_probs_train", "clip_probs_val", "clip_probs_test",
                "clip_label_embeds"
            ]
            tensor_long_keys = [
                "train_labels", "val_labels", "test_labels",
                "clip_argmax_train", "clip_argmax_val", "clip_argmax_test",
                "train_idx_from_train50k", "val_idx_from_train50k"
            ]
            for k in tensor_fp32_keys:
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(torch.float32)
            for k in tensor_long_keys:
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = data[k].long()
            print("[CACHE] Loaded successfully.")
            return data
        except Exception as e:
            print(f"[CACHE] Failed to load cache ({e}). Rebuilding...")

    print("[CACHE] Building CLIP features with strict 45k/5k/10k split...")
    t0 = time.time()

    # 读取原始数据
    train_data_all, train_label_all, test_data, test_label, class_names = read_data_cifar_100(root)
    C = len(class_names)
    assert C == 100, f"Expect 100 classes, got {C}"

    # 严格验证划分（每类 50）
    train_idx, val_idx = stratified_val_split_per_class(train_label_all, k_per_class=50, seed=seed)
    # 构建三份实际用于编码的数据
    train_data = train_data_all[train_idx]
    train_label = train_label_all[train_idx]
    val_data = train_data_all[val_idx]
    val_label = train_label_all[val_idx]

    # 变换 & 模型
    transform = get_transform(image_size=image_size)
    clip_model, _ = clip.load(model_name, device=device)
    clip_model = clip_model.eval()
    print(f"[CLIP] Loaded {model_name} on {device}.")

    # 文本标签嵌入（按 meta 顺序）
    label_embeds = build_clip_label_embedding(clip_model, class_names).to(device)

    # DataLoaders
    train_loader = DataLoader(CIFARArrayDataset(train_data, train_label, transform),
                              batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(CIFARArrayDataset(val_data,   val_label,   transform),
                              batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(CIFARArrayDataset(test_data,  test_label,  transform),
                              batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 编码三份
    train_feats, train_labels, clip_probs_train, clip_argmax_train = encode_split(clip_model, train_loader, label_embeds, "train")
    val_feats,   val_labels,   clip_probs_val,   clip_argmax_val   = encode_split(clip_model, val_loader,   label_embeds, "val")
    test_feats,  test_labels,  clip_probs_test,  clip_argmax_test  = encode_split(clip_model, test_loader,  label_embeds, "test")

    # 校验数量与均衡
    def _check_counts(name, labels_t, expect_per_class):
        arr = labels_t.numpy()
        uniq, cnts = np.unique(arr, return_counts=True)
        assert uniq.size == C, f"{name}: class count {uniq.size} != {C}"
        if expect_per_class is not None:
            bad = [int(u) for u, c in zip(uniq, cnts) if c != expect_per_class]
            assert len(bad) == 0, f"{name}: not balanced for classes {bad}"

    _check_counts("VAL", val_labels, expect_per_class=50)        # 50/类
    _check_counts("TRAIN", train_labels, expect_per_class=450)   # 450/类
    assert train_labels.numel() == 45000 and val_labels.numel() == 5000
    assert test_labels.numel() == 10000

    # 保存
    data = {
        # 特征与标签
        "train_feats": train_feats, "train_labels": train_labels,
        "val_feats":   val_feats,   "val_labels":   val_labels,
        "test_feats":  test_feats,  "test_labels":  test_labels,

        # CLIP 文本标签嵌入
        "clip_label_embeds": label_embeds.detach().cpu().to(torch.float32),  # (C,D)

        # CLIP 概率与 argmax
        "clip_probs_train": clip_probs_train, "clip_argmax_train": clip_argmax_train,
        "clip_probs_val":   clip_probs_val,   "clip_argmax_val":   clip_argmax_val,
        "clip_probs_test":  clip_probs_test,  "clip_argmax_test":  clip_argmax_test,

        # 划分信息（相对 5 万训练样本的索引）
        "train_idx_from_train50k": torch.from_numpy(train_idx).long(),
        "val_idx_from_train50k":   torch.from_numpy(val_idx).long(),
        "seed": int(seed),
        "n_val_per_class": 50,

        # 元信息
        "class_names": class_names,
        "image_size": int(image_size),
        "model_name": model_name,
        "split_note": "Strict per-class split: train=450, val=50 per class (from the 50k training set). Test=10k official.",
    }
    torch.save(data, cache_path)
    dt = time.time() - t0
    print(f"[CACHE] Saved to {cache_path} (time: {dt/60:.1f} min)")
    return data


# -----------------------
# main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(dataset_root("CIFAR100")), help="CIFAR100 python root (contains train/test/meta)")
    p.add_argument("--cache", type=str, default=str(clip_cache("CIFAR100")))
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="ViT-L/14")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--force_rebuild", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    build_or_load_clip_cache_strict_split(
        root=args.root,
        cache_path=args.cache,
        image_size=args.image_size,
        batch_size=args.batch_size,
        model_name=args.model_name,
        seed=args.seed,
        num_workers=args.num_workers,
        force_rebuild=args.force_rebuild,
    )
