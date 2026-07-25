# -*- coding: utf-8 -*-
"""
extract_CLIP_eurosat.py

- 用 CLIP (ViT-L/14) 提取 EuroSAT (RGB) 特征并缓存
- 按类别分层划分 train/val/test（默认 80%/10%/10%，可改）
- 保存：特征、标签、CLIP 文本标签嵌入、CLIP 概率与 argmax、以及可复现实验的随机种子与分割说明
- 产出的缓存键与 extract_CLIP_cifar100.py 对齐：train/val/test 的 feats/labels，
  clip_label_embeds，clip_probs_*，clip_argmax_*，并包含 class_names/image_size/model_name/split_note 等元信息
"""

import os, time, math, argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import clip
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import RAW_ROOT, clip_cache

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# 通用
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def get_transform(image_size=224):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])


def article(name: str) -> str:
    return "an" if name[0].lower() in "aeiou" else "a"


def processed_name(name: str, rm_dot=False) -> str:
    # 与 CIFAR 版保持一致的名称处理
    res = name.replace("_", " ").replace("/", " or ").replace("-", " ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


# -----------------------
# EuroSAT 数据集（RGB）
# -----------------------
class EuroSATWrapper(Dataset):
    """把 torchvision.datasets.EuroSAT 包一层，只负责 __getitem__ 输出 (PIL->tensor, label)"""
    def __init__(self, base_ds: torchvision.datasets.EuroSAT, transform):
        self.base = base_ds
        self.transform = transform
        # 兼容性：torchvision EuroSAT 通常有 .targets / .classes
        if hasattr(base_ds, "targets"):
            self.targets = np.array(base_ds.targets, dtype=np.int64)
        else:
            # 兜底：遍历一次
            self.targets = np.array([base_ds[i][1] for i in range(len(base_ds))], dtype=np.int64)
        self.classes = list(getattr(base_ds, "classes", []))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        img = self.transform(img)
        return img, int(y)


# -----------------------
# 文本标签嵌入（和 CIFAR 版一致：简单 prompt ensemble）
# -----------------------
SINGLE_TEMPLATE = ["a photo of a {}."]


@torch.no_grad()
def build_clip_label_embedding(model, categories: List[str]) -> torch.Tensor:
    templates = SINGLE_TEMPLATE
    run_on_gpu = torch.cuda.is_available()
    if run_on_gpu:
        model = model.cuda()

    label_embeds = []
    for cat in categories:
        texts = [
            tmpl.format(processed_name(cat, rm_dot=True), article=article(cat))
            for tmpl in templates
        ]
        # 与 CIFAR 脚本保持：若以 a/the 开头则前缀 "This is "
        texts = ["This is " + t if t.startswith(("a", "the")) else t for t in texts]
        tokens = clip.tokenize(texts)
        if run_on_gpu:
            tokens = tokens.cuda()
        text_emb = model.encode_text(tokens)           # (T,D)
        text_emb = F.normalize(text_emb, dim=-1)
        text_emb = F.normalize(text_emb.mean(dim=0), dim=0)  # (D,)
        label_embeds.append(text_emb)
    label_embeds = torch.stack(label_embeds, dim=0)   # (C,D)
    return label_embeds.to(torch.float32)


# -----------------------
# 分层划分（按类别）
# -----------------------
def stratified_split_by_ratio(labels: np.ndarray,
                              val_ratio: float = 0.1,
                              test_ratio: float = 0.1,
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 train_idx, val_idx, test_idx（互不相交），每类按比例抽样"""
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1
    set_seed(seed)
    labels = labels.astype(np.int64)
    classes = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(seed)

    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test
        if n_train <= 0:
            # 极端少样本容错：把 test 下调
            n_test = max(1, n_test - 1)
            n_train = n - n_val - n_test
        assert n_train > 0, f"class {c} too few samples"

        val_idx.append(idx_c[:n_val])
        test_idx.append(idx_c[n_val:n_val + n_test])
        train_idx.append(idx_c[n_val + n_test:])

    train_idx = np.concatenate(train_idx, axis=0)
    val_idx = np.concatenate(val_idx, axis=0)
    test_idx = np.concatenate(test_idx, axis=0)
    return train_idx, val_idx, test_idx


# -----------------------
# 编码一个 split
# -----------------------
@torch.no_grad()
def encode_split(clip_model, loader, label_embeds, name="train"):
    feats_list, labels_list, probs_list, argmax_list = [], [], [], []
    total = len(loader.dataset)
    seen = 0
    for xb, yb in loader:
        xb = xb.to(device)
        f = clip_model.encode_image(xb)        # (B,D)
        f = f.to(torch.float32)
        f = F.normalize(f, dim=-1)
        logits_clip = f @ label_embeds.t()     # (B,C)
        probs_clip = F.softmax(logits_clip, dim=-1).to(torch.float32)
        yhat = probs_clip.argmax(dim=-1)

        feats_list.append(f.cpu())
        labels_list.append(yb.clone())
        probs_list.append(probs_clip.cpu())
        argmax_list.append(yhat.cpu())

        seen += xb.size(0)
        if seen % 2048 == 0 or seen == total:
            print(f"[{name}] encoded {seen}/{total}")

    feats  = torch.cat(feats_list,  dim=0).to(torch.float32)
    labels = torch.cat(labels_list, dim=0).long()
    probs  = torch.cat(probs_list,  dim=0).to(torch.float32)
    argmx  = torch.cat(argmax_list, dim=0).long()
    return feats, labels, probs, argmx


# -----------------------
# 主流程：构建或加载缓存
# -----------------------
@torch.no_grad()
def build_or_load_clip_cache_eurosat(
    root: str = str(RAW_ROOT),
    cache_path: str = str(clip_cache("EuroSAT")),
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 2,
    model_name: str = "ViT-L/14",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    force_rebuild: bool = False,
) -> Dict[str, torch.Tensor]:

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if (not force_rebuild) and os.path.exists(cache_path):
        try:
            print(f"[CACHE] Loading from: {cache_path}")
            data = torch.load(cache_path, map_location="cpu")
            # 统一 dtypes（对齐 CIFAR 版）
            fp32_keys = [
                "train_feats","val_feats","test_feats",
                "clip_probs_train","clip_probs_val","clip_probs_test",
                "clip_label_embeds"
            ]
            i64_keys = [
                "train_labels","val_labels","test_labels",
                "clip_argmax_train","clip_argmax_val","clip_argmax_test",
            ]
            for k in fp32_keys:
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(torch.float32)
            for k in i64_keys:
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = data[k].long()
            print("[CACHE] Loaded successfully.")
            return data
        except Exception as e:
            print(f"[CACHE] Failed to load existing cache ({e}), will rebuild.")

    t0 = time.time()
    set_seed(seed)

    # 1) 下载/读取 EuroSAT (RGB)（torchvision 自带，含 classes/targets）
    base = torchvision.datasets.EuroSAT(
        root=root, download=True, transform=None  # 先不变换
    )
    transform = get_transform(image_size=image_size)
    ds = EuroSATWrapper(base, transform)
    labels_all = ds.targets  # (N,)
    class_names = ds.classes
    C = len(class_names)
    N = len(ds)
    print(f"[EuroSAT] N={N}, C={C} | classes={class_names}")

    # 2) 分层划分
    train_idx, val_idx, test_idx = stratified_split_by_ratio(
        labels_all, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    train_set = Subset(ds, train_idx.tolist())
    val_set   = Subset(ds, val_idx.tolist())
    test_set  = Subset(ds, test_idx.tolist())

    # 3) 模型与文本标签嵌入
    clip_model, _ = clip.load(model_name, device=device)
    clip_model = clip_model.eval()
    print(f"[CLIP] Loaded {model_name} on {device}.")
    label_embeds = build_clip_label_embedding(clip_model, class_names).to(device)  # (C,D)

    # 4) DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # 5) 编码三个 split
    train_feats, train_labels, clip_probs_train, clip_argmax_train = encode_split(clip_model, train_loader, label_embeds, "train")
    val_feats,   val_labels,   clip_probs_val,   clip_argmax_val   = encode_split(clip_model, val_loader,   label_embeds, "val")
    test_feats,  test_labels,  clip_probs_test,  clip_argmax_test  = encode_split(clip_model, test_loader,  label_embeds, "test")

    # 6) 保存（键名与 CIFAR 版保持一致）
    split_note = f"Stratified per-class split with ratios train={1.0-val_ratio-test_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f} (seed={seed})."
    data = {
        # 特征与标签
        "train_feats": train_feats, "train_labels": train_labels,
        "val_feats":   val_feats,   "val_labels":   val_labels,
        "test_feats":  test_feats,  "test_labels":  test_labels,

        # CLIP 文本标签嵌入
        "clip_label_embeds": label_embeds.detach().cpu().to(torch.float32),

        # CLIP 概率与 argmax
        "clip_probs_train": clip_probs_train, "clip_argmax_train": clip_argmax_train,
        "clip_probs_val":   clip_probs_val,   "clip_argmax_val":   clip_argmax_val,
        "clip_probs_test":  clip_probs_test,  "clip_argmax_test":  clip_argmax_test,

        # 元信息
        "class_names": class_names,
        "image_size": int(image_size),
        "model_name": model_name,
        "seed": int(seed),
        "split_note": split_note,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)
    print(f"[CACHE] Saved to {cache_path} | time={(time.time()-t0)/60:.1f} min")
    return data


# -----------------------
# main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(RAW_ROOT), help="原始数据根目录")
    p.add_argument("--cache", type=str, default=str(clip_cache("EuroSAT")))
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--model_name", type=str, default="ViT-L/14")
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_rebuild", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_or_load_clip_cache_eurosat(
        root=args.root,
        cache_path=args.cache,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
    )
