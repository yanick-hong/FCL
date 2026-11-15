# -*- coding: utf-8 -*-
"""
train_ce_fullsup_CIFAR100.py

- 完全监督（Fully Supervised）基线：使用全部训练样本的真实标签在缓存特征上训练线性分类头（仅 CE）
- 读取 extract_CLIP_cifar100.py 生成的严格 45k/5k/10k 缓存（train/val/test 特征与标签）
- 仅训练线性头（CLIP 已冻结于缓存阶段）
- 训练：P×K 均衡采样 + CrossEntropyLoss + OneCycleLR
- 验证：val_loss 与 val_acc；支持早停（loss/acc 二选一）
- 记录：每轮把 val_loss/val_acc 写入 ./val/CIFAR100/val_metrics_fullsup.csv，
        并绘制 val_acc_fullsup.png / val_loss_fullsup.png

用法示例：
    python train_ce_fullsup_CIFAR100.py \
        --cache ./cache/CIFAR100/clip_ViTL14.pt \
        --P 16 --K 16 --lr 1e-3 --lr_max 5e-3 \
        --max_epochs 200 --early_metric acc --patience 10
"""

import os
import csv
import time
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# -----------------------
# 读取严格 45k/5k/10k 缓存
# -----------------------
def load_clip_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
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


# -----------------------
# P×K 均衡采样（按真实标签）
# -----------------------
class PKBatchSampler(Sampler[List[int]]):
    """
    每个 batch 采样 P 个类别、每类 K 个样本 -> batch_size = P*K
    这里使用 *真实标签* 做类别均衡
    """
    def __init__(self, labels: torch.Tensor, P: int, K: int, seed: int = 42):
        super().__init__(None)
        self.labels = labels.cpu().numpy()
        self.P = int(P)
        self.K = int(K)
        self.seed = int(seed)

        self.class_to_indices = {}
        for c in np.unique(self.labels):
            idx = np.where(self.labels == c)[0].tolist()
            self.class_to_indices[int(c)] = idx

        self.classes = list(self.class_to_indices.keys())
        self.N = len(self.labels)
        self.batch_size = self.P * self.K
        self.steps_per_epoch = max(1, self.N // self.batch_size)

        import random
        self.ptrs = {}
        for c, idxs in self.class_to_indices.items():
            rng = random.Random(self.seed + c)
            rng.shuffle(idxs)
            self.ptrs[c] = {"idxs": idxs, "p": 0, "rng": rng}

    def __len__(self):
        return self.steps_per_epoch

    def _next_k(self, c, K):
        buf = []
        state = self.ptrs[c]
        idxs, p, rng = state["idxs"], state["p"], state["rng"]
        for _ in range(K):
            if p >= len(idxs):
                rng.shuffle(idxs)
                p = 0
            buf.append(idxs[p])
            p += 1
        state["p"] = p
        return buf

    def __iter__(self):
        import random
        rng = random.Random(self.seed)
        for _ in range(self.steps_per_epoch):
            chosen = rng.sample(self.classes, self.P) if len(self.classes) >= self.P \
                     else [rng.choice(self.classes) for _ in range(self.P)]
            batch_idx = []
            for c in chosen:
                batch_idx.extend(self._next_k(c, self.K))
            yield batch_idx


# -----------------------
# 线性头
# -----------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        return self.fc(x)


# -----------------------
# 评估 & 早停
# -----------------------
@torch.no_grad()
def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048) -> Tuple[float, float]:
    head.eval()
    ds = TensorDataset(feats, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ce = nn.CrossEntropyLoss(reduction="mean")
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in dl:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device)
        logits = head(xb)
        loss = ce(logits, yb)
        pred = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((pred == yb).sum().item())
        total += xb.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


class EarlyStopping:
    def __init__(self, metric: str = "acc", patience: int = 10, min_delta: float = 0.0):
        assert metric in ("loss", "acc")
        self.metric = metric
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf") if metric == "loss" else -float("inf")
        self.num_bad = 0

    def improved(self, val_loss: float, val_acc: float) -> bool:
        cur = val_loss if self.metric == "loss" else val_acc
        if self.metric == "loss":
            return cur < self.best - self.min_delta
        else:
            return cur > self.best + self.min_delta

    def step(self, val_loss: float, val_acc: float) -> bool:
        if self.improved(val_loss, val_acc):
            self.best = val_loss if self.metric == "loss" else val_acc
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            print(f"早停已用耐心:{self.num_bad}")
            return self.num_bad >= self.patience


# -----------------------
# 记录/绘图
# -----------------------
def _save_val_curves(epochs, losses, accs, save_dir: str, suffix: str = "fullsup"):
    os.makedirs(save_dir, exist_ok=True)
    # val_acc（%）
    plt.figure()
    plt.plot(epochs, [a * 100.0 for a in accs], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val Acc (%)")
    plt.title("Validation Accuracy (Full Supervised)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"val_acc_{suffix}.png"), dpi=150)
    plt.close()

    # val_loss
    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val CE Loss")
    plt.title("Validation Loss (Full Supervised)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"val_loss_{suffix}.png"), dpi=150)
    plt.close()


# -----------------------
# 训练主流程（完全监督）
# -----------------------
def train(
    cache_path: str = "./cache/CIFAR100/clip_ViTL14.pt",
    P: int = 16,
    K: int = 16,                 # batch_size = P*K
    lr: float = 1e-3,
    lr_max: float = 5e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    print_every: int = 20,
    ckpt_path: str = "./outputs/CIFAR100/head_ce_fullsup.pt",
    best_ckpt_path: str = "./outputs/CIFAR100/head_ce_fullsup_best.pt",
    early_metric: str = "acc",
    patience: int = 10,
    min_delta: float = 0.0,
):
    set_seed(42)

    # 0) 载入缓存
    cache = load_clip_cache_strict(cache_path)
    train_feats = cache["train_feats"].to(torch.float32)
    val_feats   = cache["val_feats"].to(torch.float32)
    train_labels = cache["train_labels"].long()
    val_labels   = cache["val_labels"].long()

    # 单位化特征（与现有脚本保持一致）
    train_feats = F.normalize(train_feats, dim=-1)
    val_feats   = F.normalize(val_feats, dim=-1)

    N, D = train_feats.shape
    C = int(max(train_labels.max().item(), val_labels.max().item())) + 1
    print(f"[INFO] Ntrain={N}, Nval={val_labels.numel()}, D={D}, C={C} | P={P}, K={K}, batch={P*K}")

    # 数据集/采样器（使用真实标签做均衡）
    train_ds = TensorDataset(train_feats, train_labels)
    batch_sampler = PKBatchSampler(labels=train_labels, P=P, K=K, seed=42)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    # 线性头
    head = LinearHead(D, C).to(device).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_max,
        total_steps=max(1, steps_per_epoch * max_epochs),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=10.0
    )

    stopper = EarlyStopping(metric=early_metric, patience=patience, min_delta=min_delta)
    ce_criterion = nn.CrossEntropyLoss(reduction="mean")

    # === 记录与输出目录 ===
    val_save_dir = "./val/CIFAR100/fullsup"
    os.makedirs(val_save_dir, exist_ok=True)
    csv_path = os.path.join(val_save_dir, "val_metrics_fullsup.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_loss", "val_acc"])  # val_acc 为 0~1
    val_epochs, val_losses, val_accs = [], [], []

    print("[TRAIN] Start training (Fully Supervised, CE only) ...")

    best_for_save = float("inf") if early_metric == "loss" else -float("inf")

    for epoch in range(1, max_epochs + 1):
        head.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0

        for ib, batch in enumerate(train_loader, start=1):
            f, y = batch
            f = f.to(device, dtype=torch.float32)
            y = y.to(device)

            logits = head(f)
            loss = ce_criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            seen += f.size(0)

            if ib % print_every == 0:
                print(
                    f"[Epoch {epoch:03d}] batch {ib:04d}/{len(train_loader):04d} | "
                    f"seen={seen:6d} | loss(avg so far)={running_loss/ib:.5f}"
                )

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] time={dt:.1f}s | avg_loss={running_loss/max(1,len(train_loader)):.5f}")

        # 验证与早停
        val_loss, val_acc = evaluate(head, val_feats, val_labels, batch_size=2048)
        print(f"[VAL] epoch={epoch:03d} | val_loss={val_loss:.5f} | val_acc={val_acc*100:.2f}%")

        # 记录、写入 CSV、刷新曲线
        val_epochs.append(epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.6f}"])
        _save_val_curves(val_epochs, val_losses, val_accs, val_save_dir, suffix="fullsup")
        print(f"[VAL-SAVE] CSV -> {csv_path} | PNG -> {val_save_dir}/val_acc_fullsup.png, val_loss_fullsup.png")

        # 保存当前与最佳
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(head.state_dict(), ckpt_path)

        cur_metric = val_loss if early_metric == "loss" else val_acc
        is_better = (cur_metric < best_for_save) if early_metric == "loss" else (cur_metric > best_for_save)
        if is_better and best_ckpt_path:
            best_for_save = cur_metric
            torch.save(head.state_dict(), best_ckpt_path)
            print(f"[CKPT] >>> Best ({early_metric}) saved to {best_ckpt_path}")

        # 早停判断
        if stopper.step(val_loss, val_acc):
            print(f"[EARLY STOP] No improvement in {early_metric} for {stopper.patience} epochs. Stop.")
            break

    print("[DONE] Training finished (Fully Supervised).")


# -----------------------
# main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default="./cache/CIFAR100/clip_ViTL14.pt")
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_max", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=200)

    # 日志/保存/早停
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--ckpt", type=str, default="./outputs/CIFAR100/temp.pt")
    p.add_argument("--best_ckpt", type=str, default="./outputs/CIFAR100/head_ce_fullsup_best.pt")
    p.add_argument("--early_metric", type=str, default="acc", choices=["loss", "acc"])
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(
        cache_path=args.cache,
        P=args.P, K=args.K,
        lr=args.lr, lr_max=args.lr_max,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        print_every=args.print_every,
        ckpt_path=args.ckpt,
        best_ckpt_path=args.best_ckpt,
        early_metric=args.early_metric,
        patience=args.patience,
        min_delta=args.min_delta,
    )
