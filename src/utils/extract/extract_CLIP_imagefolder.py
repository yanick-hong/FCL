"""Extract a reusable OpenAI CLIP cache from an ImageFolder dataset.

This covers the already extracted EuroSAT, DTD and Caltech-101 archives.  It
keeps the cache schema compatible with ``train_auc_ce.py`` and performs a
deterministic per-class train/validation/test split.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class Samples(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def split_indices(labels, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)
    train, val, test = [], [], []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * 0.1))
        n_test = max(1, round(len(indices) * 0.1))
        n_train = len(indices) - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = max(1, len(indices) - 2), 1, 1
        val.extend(indices[:n_val])
        test.extend(indices[n_val:n_val + n_test])
        train.extend(indices[n_val + n_test:n_val + n_test + n_train])
    return train, val, test


@torch.no_grad()
def encode(model, loader, text_features, device):
    features, labels, probs, argmax = [], [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        image_features = F.normalize(model.encode_image(images).float(), dim=-1)
        logits = image_features @ text_features.t()
        probabilities = logits.softmax(dim=-1)
        features.append(image_features.cpu())
        labels.append(targets.long())
        probs.append(probabilities.cpu())
        argmax.append(probabilities.argmax(dim=-1).cpu())
    return tuple(torch.cat(values) for values in (features, labels, probs, argmax))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset", choices=["eurosat", "dtd", "caltech101"], required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.path.exists(args.cache):
        print(f"[CACHE] Reusing {args.cache}")
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    source = ImageFolder(args.data_root)
    class_names = list(source.classes)
    if args.dataset == "caltech101":
        keep = [name for name in class_names if name != "BACKGROUND_Google"]
        remap = {name: i for i, name in enumerate(keep)}
        samples = [(path, remap[class_names[label]]) for path, label in source.samples if class_names[label] in remap]
        class_names = keep
    else:
        samples = list(source.samples)
    labels = [label for _, label in samples]
    train_idx, val_idx, test_idx = split_indices(labels, args.seed)
    prompts = [f"a photo of a {name.replace('_', ' ')}." for name in class_names]
    text = clip.tokenize(prompts).to(device)
    text_features = F.normalize(model.encode_text(text).float(), dim=-1)

    dataset = Samples(samples, preprocess)
    make_loader = lambda indices: DataLoader(dataset, sampler=indices, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    tr_f, tr_y, tr_p, tr_a = encode(model, make_loader(train_idx), text_features, device)
    va_f, va_y, va_p, va_a = encode(model, make_loader(val_idx), text_features, device)
    te_f, te_y, te_p, te_a = encode(model, make_loader(test_idx), text_features, device)

    output = {
        "train_feats": tr_f.float(), "train_labels": tr_y.long(),
        "val_feats": va_f.float(), "val_labels": va_y.long(),
        "test_feats": te_f.float(), "test_labels": te_y.long(),
        "clip_probs_train": tr_p.float(), "clip_argmax_train": tr_a.long(),
        "clip_probs_val": va_p.float(), "clip_argmax_val": va_a.long(),
        "clip_probs_test": te_p.float(), "clip_argmax_test": te_a.long(),
        "clip_label_embeds": text_features.cpu(), "class_names": class_names,
        "meta": {"dataset": args.dataset, "split_seed": args.seed, "source": str(Path(args.data_root).resolve())},
    }
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.cache)
    print(f"[DONE] {args.dataset}: train={len(tr_y)} val={len(va_y)} test={len(te_y)} -> {args.cache}")


if __name__ == "__main__":
    main()
