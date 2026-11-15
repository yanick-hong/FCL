# -*- coding: utf-8 -*-
"""
train_ce_CIFAR100.py
- 仅从 extract_CLIP_cifar100.py 生成的严格 45k/5k/10k 缓存读取特征与标签
- 读取 make_obs_labels.py 生成的 obs_path（y_obs, s），仅用可信样本 (s=0) 训练线性头（纯 CE）
- 用验证集 val_acc 监控并早停（只保存最佳权重）
"""

import os
import time
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clip_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
    """只读取缓存；按 extract_CLIP_cifar100.py 的字段和 dtype 整理"""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    print(f"[CACHE] Loading from: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

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


def load_obs_labels(obs_path: str) -> Dict[str, torch.Tensor]:
    """读取 make_obs_labels.py 生成的 obs_path（内含 y_obs 与 s）"""
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Obs labels not found: {obs_path}")
    data = torch.load(obs_path, map_location="cpu")
    if not ("y_obs" in data and "s" in data):
        raise KeyError(f"Obs file must contain 'y_obs' and 's'. Got keys: {list(data.keys())}")
    y_obs = data["y_obs"].long()
    s = data["s"].long()
    print(f"[OBS] Using obs_path -> y_obs(shape={tuple(y_obs.shape)}), s(shape={tuple(s.shape)})")
    return {"y_obs": y_obs, "s": s}


# -----------------------
# 只含可信样本的数据集
# -----------------------
class TrustedFeatureDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        self.f = feats
        self.y = labels
    def __len__(self): return self.f.size(0)
    def __getitem__(self, idx):
        return self.f[idx], int(self.y[idx])


# -----------------------
# 线性头
# -----------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):  # x: (B,D) float32
        return self.fc(x)


# -----------------------
# 评估（CE损失 + acc）
# -----------------------
@torch.no_grad()
def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048) -> Tuple[float, float]:
    head.eval()
    ds = TensorDataset(feats, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in dl:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device)
        logits = head(xb)
        loss = ce(logits, yb)
        pred = logits.argmax(dim=-1)
        total_loss += float(loss.item())
        total_correct += int((pred == yb).sum().item())
        total += xb.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


# -----------------------
# 早停（按 val_acc）
# -----------------------
class EarlyStoppingAcc:
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -float("inf")
        self.num_bad = 0
    def step(self, val_acc: float) -> bool:
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            print(f"已消耗耐心：{self.num_bad}")
            return self.num_bad >= self.patience


# -----------------------
# 训练主流程（纯 CE，可信样本）
# -----------------------
def train(
    cache_path: str = "./cache/CIFAR100/clip_ViTL14.pt",
    obs_labels_path: str = "./cache/CIFAR100/cifar100_obs_labels.pt",
    batch_size: int = 1024,
    lr_max: float = 5e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    print_every: int = 20,
    ckpt_best: str = "./outputs/CIFAR100/head_ce_trusted.best.pt",
    seed: int = 42,
    patience: int = 8,
):
    set_seed(seed)
    os.makedirs(os.path.dirname(ckpt_best) or ".", exist_ok=True)

    # 0) 读取缓存（严格 45k/5k/10k）
    cache = load_clip_cache_strict(cache_path)
    train_feats = cache["train_feats"].to(torch.float32)          # (45000,D)
    train_labels = cache["train_labels"].long()
    val_feats   = cache["val_feats"].to(torch.float32)            # (5000,D)
    val_labels  = cache["val_labels"].long()

    N, D = train_feats.shape
    C = int(max(train_labels.max().item(), val_labels.max().item())) + 1
    print(f"[INFO] Ntrain={N}, Nval={val_labels.numel()}, D={D}, C={C}")

    # 1) 读取 obs_path（y_obs, s），仅保留可信样本 s=0
    obs = load_obs_labels(obs_labels_path)
    y_obs_after, s_flags = obs["y_obs"], obs["s"]
    if y_obs_after.numel() != N or s_flags.numel() != N:
        raise RuntimeError(f"Obs length mismatch: feats N={N}, y_obs={y_obs_after.numel()}, s={s_flags.numel()}")
    trusted_idx = torch.nonzero(s_flags == 0, as_tuple=False).squeeze(1)
    if trusted_idx.numel() == 0:
        raise RuntimeError("No trusted samples found in obs_labels (s==0). Please check make_obs_labels.py output.")
    feats_trusted  = train_feats[trusted_idx]
    labels_trusted = y_obs_after[trusted_idx]

    # 2) DataLoader（仅可信样本）
    train_ds = TrustedFeatureDataset(feats_trusted, labels_trusted)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # 3) 线性头 + 优化器 + 调度器（正常训练）
    head = LinearHead(D, C).to(device).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr_max/10.0, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_max, total_steps=steps_per_epoch * max_epochs,
        pct_start=0.15, anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0
    )
    ce = nn.CrossEntropyLoss()

    # 4) 训练 + 验证 + 早停（按 val_acc），仅保存最佳
    stopper = EarlyStoppingAcc(patience=patience, min_delta=0.0)
    best_state = None
    best_acc = -1.0

    print("[TRAIN] Start training (pure CE on trusted samples)...")
    for epoch in range(1, max_epochs + 1):
        head.train()
        t0 = time.time()
        run_loss = run_acc = 0.0
        batches = 0
        seen = 0

        for ib, (f, y) in enumerate(train_loader, start=1):
            f = f.to(device, dtype=torch.float32)
            y = y.to(device)
            logits = head(f)
            loss = ce(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            # stats
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == y).float().mean().item()
            run_loss += float(loss.item())
            run_acc  += float(acc)
            batches  += 1
            seen     += f.size(0)

            if ib % print_every == 0:
                print(f"[Epoch {epoch:02d}] batch {ib:04d}/{len(train_loader):04d} | "
                      f"seen={seen:6d} | loss={run_loss/batches:.5f} | acc={run_acc/batches*100:.2f}%")

        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] time={dt:.1f}s | train_loss={run_loss/max(1,batches):.5f} "
              f"| train_acc={run_acc/max(1,batches)*100:.2f}%")

        # 验证与（仅）保存最佳
        val_loss, val_acc = evaluate(head, val_feats, val_labels, batch_size=2048)
        print(f"[VAL] epoch={epoch:02d} | val_loss={val_loss:.5f} | val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if ckpt_best:
                torch.save(best_state, ckpt_best)
                print(f"[CKPT] >>> Best so far (acc={best_acc*100:.2f}%) saved to {ckpt_best}")

        # 早停（按 val_acc）
        if stopper.step(val_acc):
            print(f"[EARLY STOP] val_acc has not improved for {stopper.patience} epochs. Stop.")
            break

    # 结束时确保最佳权重已落盘
    if best_state is not None and ckpt_best:
        torch.save(best_state, ckpt_best)
    print("[DONE] Training finished.")


# -----------------------
# main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default="./cache/CIFAR100/clip_ViTL14.pt")
    p.add_argument("--obs_labels_path", type=str, default="./cache/CIFAR100/cifar100_obs_labels.pt")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr_max", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=3e-3)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--ckpt_best", type=str, default="./outputs/CIFAR100/head_ce_trusted.pt")
    p.add_argument("--patience", type=int, default=8) 
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(
        cache_path=args.cache,
        obs_labels_path=args.obs_labels_path,
        batch_size=args.batch_size,
        lr_max=args.lr_max,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        print_every=args.print_every,
        ckpt_best=args.ckpt_best,
        patience=args.patience,
        seed=args.seed,
    )
