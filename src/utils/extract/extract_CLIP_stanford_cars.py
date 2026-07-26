"""Extract Stanford Cars CLIP features from the official MAT annotations."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


class CarDataset(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        return self.transform(Image.open(path).convert("RGB")), label


def scalar(value):
    while hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    return value


def read_records(annotation_file, image_dir):
    annotations = loadmat(annotation_file, squeeze_me=True, struct_as_record=False)["annotations"]
    records = []
    for annotation in annotations.ravel():
        filename = str(scalar(annotation.fname))
        # The official test annotation file contains filenames/bounding boxes
        # but intentionally omits class labels.  Test labels are not needed
        # for the noisy-label training run, so retain -1 for that split.
        class_value = getattr(annotation, "class", None)
        label = -1 if class_value is None else int(scalar(class_value)) - 1
        records.append((str(Path(image_dir) / filename), label))
    return records


def split_train(records, seed=42):
    rng = random.Random(seed)
    groups = {}
    for i, (_, label) in enumerate(records):
        groups.setdefault(label, []).append(i)
    train, val = [], []
    for label in sorted(groups):
        indices = groups[label][:]
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * 0.1))
        val.extend(indices[:n_val])
        train.extend(indices[n_val:])
    return [records[i] for i in train], [records[i] for i in val]


@torch.no_grad()
def encode(model, loader, text_features, device):
    features, labels, probs, argmax = [], [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        image_features = F.normalize(model.encode_image(images).float(), dim=-1)
        probabilities = (image_features @ text_features.t()).softmax(dim=-1)
        features.append(image_features.cpu()); labels.append(targets.long())
        probs.append(probabilities.cpu()); argmax.append(probabilities.argmax(dim=-1).cpu())
    return tuple(torch.cat(values) for values in (features, labels, probs, argmax))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if os.path.exists(args.cache):
        print(f"[CACHE] Reusing {args.cache}"); return
    root = Path(args.data_root)
    devkit = root / "devkit"
    train_records = read_records(devkit / "cars_train_annos.mat", root / "cars_train")
    test_records = read_records(devkit / "cars_test_annos.mat", root / "cars_test")
    names_raw = loadmat(devkit / "cars_meta.mat", squeeze_me=True)["class_names"]
    class_names = [str(scalar(name)) for name in names_raw.ravel()]
    train_records, val_records = split_train(train_records, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device); model.eval()
    text = clip.tokenize([f"a photo of a {name}." for name in class_names]).to(device)
    text_features = F.normalize(model.encode_text(text).float(), dim=-1)

    def load(records):
        ds = CarDataset(records, preprocess)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    tr_f, tr_y, tr_p, tr_a = encode(model, load(train_records), text_features, device)
    va_f, va_y, va_p, va_a = encode(model, load(val_records), text_features, device)
    te_f, te_y, te_p, te_a = encode(model, load(test_records), text_features, device)
    output = {
        "train_feats": tr_f.float(), "train_labels": tr_y.long(), "val_feats": va_f.float(), "val_labels": va_y.long(),
        "test_feats": te_f.float(), "test_labels": te_y.long(), "clip_probs_train": tr_p.float(), "clip_argmax_train": tr_a.long(),
        "clip_probs_val": va_p.float(), "clip_argmax_val": va_a.long(), "clip_probs_test": te_p.float(), "clip_argmax_test": te_a.long(),
        "clip_label_embeds": text_features.cpu(), "class_names": class_names,
        "meta": {"dataset": "Stanford Cars", "split_seed": args.seed, "source": str(root.resolve())},
    }
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True); torch.save(output, args.cache)
    print(f"[DONE] Stanford Cars: train={len(tr_y)} val={len(va_y)} test={len(te_y)} -> {args.cache}")


if __name__ == "__main__":
    main()
