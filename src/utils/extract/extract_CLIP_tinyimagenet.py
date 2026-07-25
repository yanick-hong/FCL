# -*- coding: utf-8 -*-
"""
extract_CLIP_tinyimagenet.py

- 从 Tiny-ImageNet-200 目录读取图像与标签
- 使用 CLIP (ViT-L/14, openai) 提取图像特征、生成文本类别嵌入并计算零样本概率
- 三个 split：
  * train: 每类 500 张中的 450 张（分层随机）
  * val  : 每类 500 张中的 50 张（来自 train 内部，分层抽取）
  * test : 官方 val/ 目录（每类 50 张，通过 val_annotations.txt 映射）
- 输出缓存字段（与 CIFAR100 版脚本保持一致命名）：
  train_feats, val_feats, test_feats                  (float32)
  train_labels, val_labels, test_labels               (int64)
  clip_label_embeds                                   (float32, shape [C, D])
  clip_probs_train, clip_probs_val, clip_probs_test   (float32, softmax 概率)
  clip_argmax_train, clip_argmax_val, clip_argmax_test(int64)
  wnids, classnames, templates                        (辅助信息)
"""

import os, csv, json, argparse, random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import open_clip  # pip install open_clip_torch
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


# -------------------- Tiny-ImageNet 读入 --------------------
def load_wnids_words(root: Path) -> Tuple[List[str], Dict[str, str]]:
    wnids_fp = root / "tiny-imagenet-200" / "wnids.txt"
    words_fp = root / "tiny-imagenet-200" / "words.txt"
    if not wnids_fp.exists():
        raise FileNotFoundError(f"Missing {wnids_fp}")
    wnids = [x.strip() for x in wnids_fp.read_text().splitlines() if x.strip()]
    # words.txt: "wnid<TAB>name1, name2, ..."
    wnid2name = {}
    if words_fp.exists():
        with open(words_fp, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    wid, names = parts[0], parts[1]
                    # 取第一个同义词作为类名
                    cname = names.split(",")[0].strip().lower()
                    wnid2name[wid] = cname
    # 回退：若无 words.txt，classname 用 wnid 代替
    for wid in wnids:
        wnid2name.setdefault(wid, wid)
    return wnids, wnid2name


def build_imagenet_templates() -> List[str]:
    # 精简版 ImageNet 模板（够用；与 openai/CLIP 常用模板同风格）
    return [
        "a photo of a {}.",
    ]

        # "a bad photo of a {}.",
        # "a photo of the {}.",
        # "a blurry photo of a {}.",
        # "a close-up photo of a {}.",
        # "a bright photo of a {}.",
        # "a cropped photo of a {}.",
        # "a black and white photo of a {}.",
        # "a low resolution photo of a {}.",
        # "a rendering of a {}.",
        # "a clean photo of a {}.",
        # "a photo of a small {}.",
        # "a photo of a large {}.",
        # "a photo of one {}.",
        # "a photo of many {}.",

class TinyImageNetTrainFolder(Dataset):
    """读取 train/，并支持分层抽样拆分 train/val_from_train"""
    def __init__(self, root: Path, wnids: List[str], split: str, per_class_val: int = 50, transform=None):
        """
        split: 'train' or 'val_from_train'
        per_class_val: 每类从 500 中拿出多少做 val_from_train（默认 50）
        """
        self.transform = transform
        self.samples = []
        self.targets = []

        train_dir = root / "tiny-imagenet-200" / "train"
        assert train_dir.exists(), f"Not found: {train_dir}"

        for cls_idx, wid in enumerate(wnids):
            img_dir = train_dir / wid / "images"
            files = [p for p in img_dir.glob("*.JPEG")]
            files.sort()
            if len(files) != 500:
                # 个别镜像可能清理了灰度图，容错处理
                pass

            # 固定打乱后切分（每类）
            rng = random.Random(2025 + cls_idx)
            files_shuf = files.copy()
            rng.shuffle(files_shuf)

            val_take = min(per_class_val, len(files_shuf) // 10)  # 约 10% 兜底
            if split == "val_from_train":
                chosen = files_shuf[:val_take]
            else:
                chosen = files_shuf[val_take:]

            self.samples.extend(chosen)
            self.targets.extend([cls_idx] * len(chosen))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[i]


class TinyImageNetValOfficial(Dataset):
    """读取 val/ 官方验证（本脚本作为 test 使用）"""
    def __init__(self, root: Path, wnids: List[str], transform=None):
        self.transform = transform
        self.samples = []
        self.targets = []

        val_dir = root / "tiny-imagenet-200" / "val"
        img_dir = val_dir / "images"
        ann_fp  = val_dir / "val_annotations.txt"
        assert img_dir.exists() and ann_fp.exists(), f"Not found: {img_dir} or {ann_fp}"

        # 解析 TSV：filename \t wnid \t (bbox...)
        wid2idx = {wid: i for i, wid in enumerate(wnids)}
        with open(ann_fp, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    fname, wid = row[0], row[1]
                    img_path = img_dir / fname
                    if img_path.exists() and wid in wid2idx:
                        self.samples.append(img_path)
                        self.targets.append(wid2idx[wid])

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[i]


# -------------------- CLIP 构建 --------------------
def build_clip(model_name="ViT-L-14", pretrained="openai", device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, preprocess, tokenizer


@torch.no_grad()
def build_text_embeds(model, tokenizer, classnames: List[str], templates: List[str], device="cuda") -> torch.Tensor:
    texts = []
    for cname in classnames:
        texts.extend([tmpl.format(cname) for tmpl in templates])
    text_tokens = tokenizer(texts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    T = len(templates); C = len(classnames)
    text_feats = text_feats.view(C, T, -1).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats.float()  # [C, D]


@torch.no_grad()
def extract_split(
    model, dataloader, text_embeds: torch.Tensor, device="cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats_list, probs_list, argmax_list = [], [], []
    logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else torch.tensor(1.0, device=device)

    for imgs, _ in tqdm(dataloader, desc="Encode"):
        imgs = imgs.to(device)
        img_feats = model.encode_image(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [B, D]
        # logits = scale * cos_sim
        logits = (logit_scale * img_feats @ text_embeds.t())
        probs = logits.softmax(dim=-1)
        feats_list.append(img_feats.float().cpu())
        probs_list.append(probs.float().cpu())
        argmax_list.append(probs.argmax(dim=-1).long().cpu())

    feats = torch.cat(feats_list, dim=0)
    probs = torch.cat(probs_list, dim=0)
    argmax = torch.cat(argmax_list, dim=0)
    return feats, probs, argmax


def run(args):
    if Path(args.out_path).expanduser().exists() and not args.force_rebuild:
        print(f"[CACHE] Reusing existing features: {Path(args.out_path).expanduser().resolve()}")
        return
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu"

    root = Path(args.data_root).expanduser().resolve()
    wnids, wnid2name = load_wnids_words(root)
    classnames = [wnid2name[w] for w in wnids]
    templates = build_imagenet_templates()

    # CLIP
    model, preprocess, tokenizer = build_clip(args.model, args.pretrained, device=device)

    # Datasets / Loaders
    ds_train = TinyImageNetTrainFolder(root, wnids, split="train",        per_class_val=args.per_class_val, transform=preprocess)
    ds_valtr = TinyImageNetTrainFolder(root, wnids, split="val_from_train", per_class_val=args.per_class_val, transform=preprocess)
    ds_test  = TinyImageNetValOfficial(root, wnids, transform=preprocess)

    def make_loader(ds):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    ld_train = make_loader(ds_train)
    ld_val   = make_loader(ds_valtr)
    ld_test  = make_loader(ds_test)

    print(f"[Info] classes={len(classnames)} | train={len(ds_train)} | val={len(ds_valtr)} | test={len(ds_test)}")

    # Text embeds
    text_embeds = build_text_embeds(model, tokenizer, classnames, templates, device=device)  # [C, D]

    # 提取三个 split
    train_feats, train_probs, train_argmax = extract_split(model, ld_train, text_embeds, device=device)
    val_feats,   val_probs,   val_argmax   = extract_split(model, ld_val,   text_embeds, device=device)
    test_feats,  test_probs,  test_argmax  = extract_split(model, ld_test,  text_embeds, device=device)

    # 标签（DataLoader 按顺序输出，与 feats 对齐）
    def collect_labels(ds):
        return torch.tensor(ds.targets, dtype=torch.long)

    train_labels = collect_labels(ds_train)
    val_labels   = collect_labels(ds_valtr)
    test_labels  = collect_labels(ds_test)

    cache = {
        # 特征与标签
        "train_feats": train_feats, "val_feats": val_feats, "test_feats": test_feats,
        "train_labels": train_labels, "val_labels": val_labels, "test_labels": test_labels,
        # 零样本概率与 argmax（供基线/加权等使用）
        "clip_probs_train": train_probs, "clip_probs_val": val_probs, "clip_probs_test": test_probs,
        "clip_argmax_train": train_argmax, "clip_argmax_val": val_argmax, "clip_argmax_test": test_argmax,
        # 文本标签嵌入
        "clip_label_embeds": text_embeds.cpu(),
        # 额外元数据
        "wnids": wnids,
        "classnames": classnames,
        "templates": templates,
        "model_name": args.model,
        "pretrained": args.pretrained,
        "split_notes": "train=450/cls from train/; val=50/cls from train/; test=official val/",
    }

    out = Path(args.out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(f"[SAVE] {out}")
    print("[DONE]")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--out_path", type=str, default=str(clip_cache("tiny-imagenet-200")))
    p.add_argument("--model", type=str, default="ViT-L-14")
    p.add_argument("--pretrained", type=str, default="openai")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--per_class_val", type=int, default=50, help="从 train/ 每类抽多少张进 val_from_train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--force_rebuild", action="store_true", help="忽略已有缓存并重新提取")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
