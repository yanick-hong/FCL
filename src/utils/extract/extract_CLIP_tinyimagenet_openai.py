"""Extract Tiny-ImageNet features with the locally cached OpenAI CLIP model."""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class TinyDataset(Dataset):
    def __init__(self, records, transform): self.records, self.transform = records, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        path, label = self.records[i]
        return self.transform(Image.open(path).convert("RGB")), label


def read_classes(root):
    base = Path(root) / "tiny-imagenet-200"
    wnids = [x.strip() for x in (base / "wnids.txt").read_text().splitlines() if x.strip()]
    names = {w: w for w in wnids}
    words = base / "words.txt"
    if words.exists():
        for line in words.read_text().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2: names[parts[0]] = parts[1].split(",")[0].strip()
    return base, wnids, [names[w] for w in wnids]


def train_records(base, wnids, per_class_val, seed):
    train, val = [], []
    for cls, wnid in enumerate(wnids):
        paths = sorted((base / "train" / wnid / "images").glob("*.JPEG"))
        rng = random.Random(seed + cls); rng.shuffle(paths)
        val.extend((str(p), cls) for p in paths[:per_class_val])
        train.extend((str(p), cls) for p in paths[per_class_val:])
    return train, val


def test_records(base, wnids):
    lookup = {w: i for i, w in enumerate(wnids)}
    records = []
    with (base / "val" / "val_annotations.txt").open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2 and row[1] in lookup:
                records.append((str(base / "val" / "images" / row[0]), lookup[row[1]]))
    return records


@torch.no_grad()
def encode(model, loader, text_features, device):
    fs, ys, ps, aa = [], [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        f = F.normalize(model.encode_image(images).float(), dim=-1)
        p = (f @ text_features.t()).softmax(dim=-1)
        fs.append(f.cpu()); ys.append(labels.long()); ps.append(p.cpu()); aa.append(p.argmax(dim=-1).cpu())
    return tuple(torch.cat(x) for x in (fs, ys, ps, aa))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True); p.add_argument("--cache", required=True)
    p.add_argument("--batch_size", type=int, default=256); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42); a = p.parse_args()
    if Path(a.cache).exists(): print(f"[CACHE] Reusing {a.cache}"); return
    base, wnids, names = read_classes(a.data_root)
    train, val = train_records(base, wnids, 50, a.seed); test = test_records(base, wnids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device); model.eval()
    tokens = clip.tokenize([f"a photo of a {n}." for n in names]).to(device)
    text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
    def loader(records): return DataLoader(TinyDataset(records, preprocess), batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)
    tr_f, tr_y, tr_p, tr_a = encode(model, loader(train), text_features, device)
    va_f, va_y, va_p, va_a = encode(model, loader(val), text_features, device)
    te_f, te_y, te_p, te_a = encode(model, loader(test), text_features, device)
    out = {"train_feats": tr_f, "train_labels": tr_y, "val_feats": va_f, "val_labels": va_y, "test_feats": te_f, "test_labels": te_y, "clip_probs_train": tr_p, "clip_argmax_train": tr_a, "clip_probs_val": va_p, "clip_argmax_val": va_a, "clip_probs_test": te_p, "clip_argmax_test": te_a, "clip_label_embeds": text_features.cpu(), "classnames": names, "wnids": wnids, "meta": {"dataset": "Tiny-ImageNet-200", "source": str(base.resolve()), "split_seed": a.seed}}
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True); torch.save(out, a.cache)
    print(f"[DONE] Tiny-ImageNet: train={len(tr_y)} val={len(va_y)} test={len(te_y)} -> {a.cache}")


if __name__ == "__main__": main()
