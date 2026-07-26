# -*- coding: utf-8 -*-
"""
extract_CLIP_food101.py

- 提取 Food-101 的 CLIP ViT-L/14 特征缓存，保证与 CIFAR100 缓存字段一致

python src/utils/extract/extract_CLIP_food101.py \
  --data_root "${DATA_ROOT}/raw/food-101" \
  --out "${DATA_ROOT}/processed/food101_clip_vit-l-14_embeddings.pt" \
  --model "ViT-L/14" \
  --batch 256 --workers 8 \
  --val-per-class 100 --seed 42 --download

"""

import os, argparse, time, random
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101
import clip
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import clip_cache, dataset_root

# -----------------------------------------------------------
# 实用函数
# -----------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _build_text_embeds(model, device, classnames: List[str]) -> torch.Tensor:
    """使用单模板 'a photo of a {}' 生成文本特征；返回 (C, D) 已单位化且为 float32。"""
    template = "a photo of a {}"
    texts = [template.format(name.replace('_', ' ')) for name in classnames]
    with torch.no_grad():
        tokens = clip.tokenize(texts).to(device)
        text_feats = model.encode_text(tokens)
        text_feats = text_feats.float()              # 强制 fp32
        text_feats = F.normalize(text_feats, dim=-1)
    return text_feats  # (C, D) float32

def _extract_split(
    model, preprocess, device,
    dataset: Food101, indices: List[int],
    text_embeds: torch.Tensor, batch_size: int, workers: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从给定索引的子集提取：图像特征 feats (N, D), 概率 clip_probs(N, C), 标签 labels(N,).
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    feats_list, probs_list, labels_list = [], [], []
    te = text_embeds.to(device=device, dtype=torch.float32)  # 确保 fp32
    with torch.no_grad():
        for imgs, ys in loader:
            imgs = imgs.to(device)
            ys = ys.to(device, non_blocking=True).long()
            img_feats = model.encode_image(imgs).float()      # 强制 fp32
            img_feats = F.normalize(img_feats, dim=-1)        # (B, D) float32

            # Zero-shot：logits = scale * image @ text^T
            logits = 100.0 * (img_feats @ te.t())             # (B, C) float32
            probs = logits.softmax(dim=-1).float()            # (B, C)

            feats_list.append(img_feats.cpu().float())
            probs_list.append(probs.cpu().float())
            labels_list.append(ys.cpu().long())

    feats = torch.cat(feats_list, dim=0).contiguous().float()
    probs = torch.cat(probs_list, dim=0).contiguous().float()
    labels = torch.cat(labels_list, dim=0).contiguous().long()
    assert feats.shape[0] == probs.shape[0] == labels.shape[0]
    return feats, probs, labels

def _stratified_val_indices(dataset: Food101, per_class: int, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    在官方 train 划分上，按每类抽取 per_class 个样本作为验证集（stratified）。
    返回 train_indices, val_indices（相对于 dataset 的索引）。
    """
    label_buckets: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        label_buckets[int(y)].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for _, idxs in label_buckets.items():
        rng.shuffle(idxs)
        v = min(per_class, len(idxs)) if per_class > 0 else 0
        val_take = idxs[:v]
        train_take = idxs[v:]
        val_idx.extend(val_take)
        train_idx.extend(train_take)
    return train_idx, val_idx

# -----------------------------------------------------------
# 主流程
# -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(dataset_root("food-101")), help="Food-101 数据根目录（含 images/、meta/ 等）")
    ap.add_argument("--out", type=str, default=str(clip_cache("food101")), help="输出 .pt 路径")
    ap.add_argument("--model", type=str, default="ViT-L/14", help="CLIP 模型名，如 'ViT-L/14', 'ViT-B/32'")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-per-class", type=int, default=100, help="从官方 train 中每类抽取多少样本作为验证集")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download", action="store_true", help="若本地不存在则自动下载 Food-101")
    ap.add_argument("--force_rebuild", action="store_true", help="忽略已有缓存并重新提取")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force_rebuild:
        print(f"[CACHE] Reusing existing features: {args.out}")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # 1) 加载 CLIP 模型 & 预处理
    print(f"[CLIP] Loading model: {args.model}")
    model, preprocess = clip.load(args.model, device=device, jit=False)
    model.eval()

    # 2) 加载 Food-101 官方划分
    # torchvision expects ``root/food-101/{images,meta}``, while our project
    # convention stores the extracted dataset itself at
    # ``data/raw/food-101/{images,meta}``.  Accept both forms.
    data_root = Path(args.data_root)
    if (data_root / "images").is_dir() and (data_root / "meta").is_dir():
        data_root = data_root.parent
    print(f"[DATA] Loading Food-101 from: {data_root} | download={args.download}")
    ds_train_for_split = Food101(root=data_root, split='train', transform=None, download=args.download)
    ds_train = Food101(root=data_root, split='train', transform=preprocess, download=False)
    ds_test  = Food101(root=data_root, split='test',  transform=preprocess, download=False)

    # 3) 类名顺序
    classnames = list(ds_train.classes)
    C = len(classnames)
    print(f"[DATA] Classes: {C}")

    # 4) 文本嵌入 (C, D) float32
    print("[CLIP] Building text embeddings ...")
    text_embeds = _build_text_embeds(model, device, classnames)  # float32

    # 5) 划分验证集
    train_idx, val_idx = _stratified_val_indices(ds_train_for_split, per_class=args.val_per_class, seed=args.seed)
    print(f"[SPLIT] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(ds_test)}")

    # 6) 提取特征
    t0 = time.time()
    print("[EXTRACT] Train split ...")
    train_feats, clip_probs_train, train_labels = _extract_split(
        model, preprocess, device, ds_train, train_idx, text_embeds, args.batch, args.workers
    )
    print("[EXTRACT] Val split ...")
    val_feats, clip_probs_val, val_labels = _extract_split(
        model, preprocess, device, ds_train, val_idx, text_embeds, args.batch, args.workers
    )
    print("[EXTRACT] Test split ...")
    test_feats, clip_probs_test, test_labels = _extract_split(
        model, preprocess, device, ds_test, list(range(len(ds_test))), text_embeds, args.batch, args.workers
    )
    print(f"[TIME] Feature extraction done in {time.time()-t0:.1f}s")

    # 7) zero-shot 预测
    clip_argmax_train = clip_probs_train.argmax(dim=-1).long()
    clip_argmax_val   = clip_probs_val.argmax(dim=-1).long()
    clip_argmax_test  = clip_probs_test.argmax(dim=-1).long()

    # 8) 保存缓存（统一 fp32 / int64）
    cache = {
        "train_feats": train_feats.float(),
        "val_feats":   val_feats.float(),
        "test_feats":  test_feats.float(),

        "clip_probs_train": clip_probs_train.float(),
        "clip_probs_val":   clip_probs_val.float(),
        "clip_probs_test":  clip_probs_test.float(),

        "train_labels": train_labels.long(),
        "val_labels":   val_labels.long(),
        "test_labels":  test_labels.long(),

        "clip_argmax_train": clip_argmax_train.long(),
        "clip_argmax_val":   clip_argmax_val.long(),
        "clip_argmax_test":  clip_argmax_test.long(),

        "clip_label_embeds": text_embeds.cpu().float(),

        "classnames": classnames,
        "meta": {
            "dataset": "Food-101",
            "model": args.model,
            "val_per_class": args.val_per_class,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    torch.save(cache, args.out)
    print(f"[SAVE] Cache saved to: {args.out}")
    print("[DONE]")


if __name__ == "__main__":
    main()
