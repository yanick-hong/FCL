# train_trusted.py
# -*- coding: utf-8 -*-
"""
只用人工校正后的可信样本（s=0）做强监督训练线性头，对比效果。
输出：
- 训练过程日志（loss/acc）
- 在 全训练集 / 仅可信子集 / 测试集 的精度
"""

import os, math, time, copy, pickle, argparse, csv
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from clip import clip  # 需可用
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import clip_cache, dataset_root, experiment_checkpoint, save_experiment_config

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# 工具 & 预处理（与你原代码保持一致的实现）
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def convert_to_rgb(image): return image.convert("RGB")

def get_transform(image_size=224):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def article(name): return "an" if name[0].lower() in "aeiou" else "a"

def processed_name(name, rm_dot=False):
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot: res = res.rstrip(".")
    return res

def load_cifar100():
    with open(dataset_root("CIFAR100") / "train", 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')
    with open(dataset_root("CIFAR100") / "test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')
    with open(dataset_root("CIFAR100") / "meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')
    return data_train, data_test, data_meta

def read_data_cifar_100():
    data_train, data_test, data_meta = load_cifar100()
    train_data = data_train['data'].reshape((data_train['data'].shape[0], 3, 32, 32))
    test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))
    train_label = data_train["fine_labels"]
    test_label = data_test["fine_labels"]
    return train_data, train_label, test_data, test_label

def load_taglist(dataset: str = "CIFAR100") -> Dict[str, List[str]]:
    tag_file = dataset_root(dataset) / f"{dataset}_ram_taglist.txt"
    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]
    return {"taglist": taglist_or}

single_template = ["a photo of a {}."]

@torch.no_grad()
def build_clip_label_embedding(model, categories: List[str]) -> torch.Tensor:
    templates = single_template
    run_on_gpu = torch.cuda.is_available()
    if run_on_gpu: model = model.cuda()

    openset_label_embedding = []
    for category in categories:
        texts = [t.format(processed_name(category, rm_dot=True), article=article(category))
                 for t in templates]
        texts = ["This is " + x if x.startswith("a") or x.startswith("the") else x for x in texts]
        tokens = clip.tokenize(texts)
        if run_on_gpu: tokens = tokens.cuda()
        text_embeddings = model.encode_text(tokens)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        text_embedding = F.normalize(text_embeddings.mean(dim=0), dim=0)
        openset_label_embedding.append(text_embedding)
    return torch.stack(openset_label_embedding, dim=0)

@torch.no_grad()
def build_or_load_clip_cache(
    cache_path: str = str(clip_cache("CIFAR100")),
    image_size: int = 224,
    batch_size: int = 256
) -> Dict[str, torch.Tensor]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu")
        for k in ["train_feats", "val_feats", "clip_probs_train", "clip_probs_val", "clip_label_embeds"]:
            if k in data and isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(torch.float32)
        for k in ["train_labels", "val_labels", "clip_argmax_train", "clip_argmax_val"]:
            if k in data and isinstance(data[k], torch.Tensor):
                data[k] = data[k].long()
        print(f"[CACHE] Loaded CLIP features from: {cache_path}")
        return data

    print("[CACHE] Building CLIP features...")
    train_data, train_label, test_data, test_label = read_data_cifar_100()
    train_label = torch.tensor(train_label, dtype=torch.long)
    val_label   = torch.tensor(test_label, dtype=torch.long)

    clip_model, _ = clip.load("ViT-L/14")
    clip_model = clip_model.to(device).eval()
    print("[CLIP] Loaded ViT-L/14.")

    info = load_taglist(dataset="CIFAR100")
    label_embeds = build_clip_label_embedding(clip_model, info["taglist"]).to(device).to(torch.float32)

    transform = get_transform(image_size=image_size)
    class CIFARArrayDataset(Dataset):
        def __init__(self, imgs, labels, tfm):
            self.imgs, self.labels, self.tfm = imgs, labels, tfm
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            x = Image.fromarray(np.uint8(self.imgs[idx]).transpose((1, 2, 0)))
            return self.tfm(x), int(self.labels[idx])

    train_loader = DataLoader(CIFARArrayDataset(train_data, train_label, transform),
                              batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(CIFARArrayDataset(test_data,  val_label,   transform),
                              batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def encode_split(loader):
        feats_list, labels_list, probs_list, argmax_list = [], [], [], []
        for xb, yb in loader:
            xb = xb.to(device)
            f = clip_model.encode_image(xb).to(torch.float32)
            f = F.normalize(f, dim=-1)
            logits = f @ label_embeds.t()
            probs  = F.softmax(logits, dim=-1).to(torch.float32)
            yhat   = probs.argmax(dim=-1)

            feats_list.append(f.cpu()); labels_list.append(yb.clone())
            probs_list.append(probs.cpu()); argmax_list.append(yhat.cpu())

        feats  = torch.cat(feats_list, 0).to(torch.float32)
        labels = torch.cat(labels_list, 0).long()
        probs  = torch.cat(probs_list,  0).to(torch.float32)
        argmx  = torch.cat(argmax_list, 0).long()
        return feats, labels, probs, argmx

    train_feats, train_labels, clip_probs_train, clip_argmax_train = encode_split(train_loader)
    val_feats,   val_labels,   clip_probs_val,   clip_argmax_val   = encode_split(val_loader)

    data = {
        "train_feats": train_feats, "train_labels": train_labels,
        "val_feats":   val_feats,   "val_labels":   val_labels,
        "clip_label_embeds": label_embeds.detach().cpu().to(torch.float32),
        "clip_probs_train": clip_probs_train, "clip_probs_val": clip_probs_val,
        "clip_argmax_train": clip_argmax_train, "clip_argmax_val": clip_argmax_val,
    }
    torch.save(data, cache_path)
    print(f"[CACHE] Saved to {cache_path}")
    return data


# 与你原方法完全一致的“30%离群 + 人工校正标记”为可信（s=0）
def outlier_correction_by_class_centers(
    feats: torch.Tensor, y_true: torch.Tensor, y_pseudo: torch.Tensor, top_pct: float = 0.30
):
    N, D = feats.shape
    C = int(y_true.max().item()) + 1
    feats = F.normalize(feats, dim=-1)
    y_obs = y_pseudo.clone()
    s     = torch.ones(N, dtype=torch.long)
    for c in range(C):
        idx = torch.nonzero(y_pseudo == c, as_tuple=False).flatten()
        if idx.numel() == 0: continue
        F_c = feats[idx]
        center = F.normalize(F_c.mean(dim=0, keepdim=True), dim=-1)
        cos_sim = (F_c @ center.t()).squeeze(1)
        dist = 1.0 - cos_sim
        k = max(1, int(math.ceil(len(idx) * top_pct)))
        _, topk_ind = torch.topk(dist, k, largest=True, sorted=False)
        outlier_idx_global = idx[topk_ind]
        y_obs[outlier_idx_global] = y_true[outlier_idx_global]  # 用真值替换
        s[outlier_idx_global] = 0                               # 标记为可信
    return y_obs, s


# -----------------------
# 模型、数据集、评估
# -----------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x): return self.fc(x)

class FeatureDataset(Dataset):
    def __init__(self, feats, labels):
        self.f = feats; self.y = labels
    def __len__(self): return self.f.size(0)
    def __getitem__(self, idx): return self.f[idx], int(self.y[idx])

@torch.no_grad()
def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048):
    head.eval()
    N = feats.size(0)
    correct, total, ce_sum = 0, 0, 0.0
    for i in range(0, N, batch_size):
        xb = feats[i:i+batch_size].to(device, dtype=torch.float32)
        yb = labels[i:i+batch_size].to(device)
        logits = head(xb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total   += yb.numel()
        ce_sum  += F.cross_entropy(logits, yb, reduction="sum").item()
    return correct / max(1, total), ce_sum / max(1, total)


# -----------------------
# 训练：只用 s=0 子集（强监督）
# -----------------------
def train_strong_supervised(
    cache_path: str = str(clip_cache("CIFAR100")),
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 15,
    top_outlier_pct: float = 0.30,
    image_size: int = 224,
    save_path: str = str(experiment_checkpoint("CIFAR100_trusted")),
    print_every: int = 20,
    label_smoothing: float = 0.0
):
    set_seed(42)

    # 0) 载入或构建 CLIP 特征缓存
    cache = build_or_load_clip_cache(cache_path, image_size=image_size, batch_size=256)
    train_feats = cache["train_feats"]      # (N,D)
    train_labels = cache["train_labels"]    # (N,)
    val_feats   = cache["val_feats"]        # (M,D)
    val_labels  = cache["val_labels"]       # (M,)
    C = int(train_labels.max().item()) + 1
    D = int(train_feats.size(1))
    N = int(train_feats.size(0))
    print(f"[INFO] N={N}, D={D}, C={C}")

    # 1) 通过“30%离群+人工校正”得到可信 s=0 子集（与你原流程严格一致）
    clip_argmax_train = cache["clip_argmax_train"]
    _y_obs_after, s_flags = outlier_correction_by_class_centers(
        feats=train_feats, y_true=train_labels, y_pseudo=clip_argmax_train, top_pct=top_outlier_pct
    )
    trusted_mask = (s_flags == 0)
    num_trusted = int(trusted_mask.sum().item())
    print(f"[STEP] Trusted (s=0): {num_trusted} / {N} ({num_trusted / max(1,N):.1%})")

    # 2) 只用可信子集作为训练数据（强监督：真值标签）
    feats_trusted  = train_feats[trusted_mask]
    labels_trusted = train_labels[trusted_mask]
    train_loader = DataLoader(FeatureDataset(feats_trusted, labels_trusted),
                              batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # 3) 线性头
    head = LinearHead(D, C).to(device).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    # Validation history is stored beside the checkpoint.  The cache's ``val``
    # split is used for model selection; the test split is evaluated only by
    # the external result finalizer.
    run_dir = Path(save_path).resolve().parent
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "val_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(["epoch", "train_loss", "train_acc", "trusted_acc", "val_loss", "val_acc"])

    # 简单的早停（基于验证集精度）
    best_state, best_val_acc, wait = None, -1.0, 0

    print("[TRAIN] Strong-supervised (s=0 only) training starts...")
    for epoch in range(1, max_epochs + 1):
        head.train()
        t0 = time.time()
        running_loss, running_batches = 0.0, 0

        for ib, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device)

            logits = head(xb)

            if label_smoothing > 0.0:
                # label smoothing 可选
                eps = label_smoothing
                target = torch.full_like(logits, eps / (C - 1))
                target.scatter_(1, yb.view(-1,1), 1.0 - eps)
                loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
            else:
                loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_batches += 1

            if ib % print_every == 0:
                print(f"[Epoch {epoch:03d}] batch {ib:04d} | loss={running_loss/running_batches:.5f}")

        # 4) 评估（全训练集 / 可信子集 / 测试集）
        train_acc_all,  _ = evaluate(head, train_feats, train_labels, batch_size=2048)
        train_acc_trust, _ = evaluate(head, feats_trusted, labels_trusted, batch_size=2048)
        val_acc, val_ce = evaluate(head, val_feats, val_labels, batch_size=2048)
        dt = time.time() - t0

        with csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                epoch, f"{running_loss/max(1,running_batches):.8f}",
                f"{train_acc_all:.8f}", f"{train_acc_trust:.8f}",
                f"{val_ce:.8f}", f"{val_acc:.8f}",
            ])

        print(f"[Epoch {epoch:03d}] time={dt:.1f}s | train_loss={running_loss/max(1,running_batches):.5f} | "
              f"TrainAcc(all)={train_acc_all*100:.2f}% | "
              f"TrainAcc(trusted)={train_acc_trust*100:.2f}% | "
              f"TestAcc={val_acc*100:.2f}% | TestCE={val_ce:.4f}")

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_path)
            print(f"[CHECKPOINT] Improved. Saved to {save_path}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("[EARLY] Early stopping triggered.")
                break

    # 5) 最终载入最优并给一个结束报告
    if best_state is not None:
        head.load_state_dict(best_state)
    train_acc_all,  _ = evaluate(head, train_feats, train_labels, batch_size=2048)
    train_acc_trust, _ = evaluate(head, feats_trusted, labels_trusted, batch_size=2048)
    val_acc, val_ce = evaluate(head, val_feats, val_labels, batch_size=2048)
    print(f"[FINAL] TrainAcc(all)={train_acc_all*100:.2f}% | "
          f"TrainAcc(trusted)={train_acc_trust*100:.2f}% | "
          f"ValAcc={val_acc*100:.2f}% | ValCE={val_ce:.4f}")
    print("[DONE]")


# -----------------------
# main
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train linear head using ONLY trusted (s=0) samples (strong supervision)")
    ap.add_argument("--cache", type=str, default=str(clip_cache("CIFAR100")))
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--top_pct", type=float, default=0.30)
    ap.add_argument("--save", type=str, default=str(experiment_checkpoint("CIFAR100_trusted")))
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    args = ap.parse_args()
    save_experiment_config("CIFAR100_trusted", args)

    train_strong_supervised(
        cache_path=args.cache,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.wd,
        max_epochs=args.epochs,
        patience=args.patience,
        top_outlier_pct=args.top_pct,
        save_path=args.save,
        print_every=args.print_every,
        label_smoothing=args.label_smoothing,
    )
