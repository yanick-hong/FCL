# -*- coding: utf-8 -*-
"""
train_auc_ce_CIFAR100_new.py

- 从 extract_CLIP_cifar100.py 的严格 45k/5k/10k 缓存读取特征与标签
- 仅训练线性头（CLIP 已冻结）
- 样本划分：按 CLIP 预测类 -> 类中心 -> 余弦距离，取每类 top-30% 为“离群样本”，做人工校正：y_obs <- y_true，标记 s=0（可信）；其余 s=1（不可信）
- 条件标签分布（仅 s=1）采用 q = alpha * p_model + (1 - alpha) * p_clip（alpha 线性 warmup）
- 训练目标：
    * 热身阶段（前 burn_in_epochs）：可信（s=0）→ CE；不可信（s=1）→ 软 CE 蒸馏到 CLIP（不启用 AUC）
    * 正常阶段：可信（s=0）→ CE 锚点；不可信（s=1）→ A11 的 OVR pairwise logistic AUC（温度 tau）
  总损失： w_auc_eff * L_auc + w_ce_eff * L_ce_trusted + L_softCE_untrusted(仅热身)
- P×K 均衡采样、小批次全对配对（O(n^2)）
- 验证集 val_acc 早停
- 每个 epoch 记录 val_loss/val_acc 到 ./val/CIFAR100/val_metrics.csv，并绘制 val_acc.png / val_loss.png
"""

import os
import math
import time
import argparse
from typing import Tuple, Dict, List

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset

import matplotlib
matplotlib.use("Agg")  # 服务器/无显示环境
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
# 仅“读取”严格 45k/5k/10k 缓存
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
# 第 1 步：类中心 & 前 30% 离群人工校正（仅训练集）
# -----------------------
def outlier_correction_by_class_centers(
    feats: torch.Tensor,         # (N,D)
    y_true: torch.Tensor,        # (N,)
    y_pseudo: torch.Tensor,      # (N,)  —— CLIP argmax
    top_pct: float = 0.30
):
    N, D = feats.shape
    C = int(y_true.max().item()) + 1
    feats = F.normalize(feats, dim=-1)

    y_obs = y_pseudo.clone()
    s     = torch.ones(N, dtype=torch.long)  # 默认 s=1 不可信
    print("[STEP 1] Computing class centers and outliers ...")
    for c in range(C):
        idx = torch.nonzero(y_pseudo == c, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        F_c = feats[idx]  # (Nc,D)
        center = F.normalize(F_c.mean(dim=0, keepdim=True), dim=-1)  # (1,D)
        cos_sim = (F_c @ center.t()).squeeze(1)   # (Nc,)
        dist = 1.0 - cos_sim
        k = max(1, int(math.ceil(len(idx) * top_pct)))
        _, topk_ind = torch.topk(dist, k, largest=True, sorted=False)
        outlier_idx_global = idx[topk_ind]
        # 人工校正：用真值替换并记为可信
        y_obs[outlier_idx_global] = y_true[outlier_idx_global]
        s[outlier_idx_global] = 0
        print(f"[STEP 1] class={c:03d} | size={len(idx):4d} | corrected(outliers)={k:4d}")
    total_trusted = int((s == 0).sum().item())
    print(f"[STEP 1] Trusted (s=0): {total_trusted} | Untrusted (s=1): {N - total_trusted}")
    return y_obs, s


# -----------------------
# 特征数据集
# -----------------------
class FeatureDataset(Dataset):
    """
    返回：
      f: (D,) float32
      y_true: int
      y_obs:  int  （人工校正后的观测标签）
      s: 0/1
      p_clip: (C,) float32
    """
    def __init__(self, feats, y_true, y_obs, s, p_clip):
        self.f = feats
        self.y_true = y_true
        self.y_obs = y_obs
        self.s = s
        self.p_clip = p_clip
    def __len__(self): return self.f.size(0)
    def __getitem__(self, idx):
        return self.f[idx], int(self.y_true[idx]), int(self.y_obs[idx]), int(self.s[idx]), self.p_clip[idx]


# -----------------------
# P×K 均衡采样 BatchSampler
# -----------------------
class PKBatchSampler(Sampler[List[int]]):
    """
    每个 batch 采样 P 个类别、每类 K 个样本 -> batch_size = P*K
    这里用 y_obs 做类别均衡
    """
    def __init__(self, labels_obs: torch.Tensor, P: int, K: int, seed: int = 42):
        super().__init__(None)
        self.labels = labels_obs.cpu().numpy()
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
    def forward(self, x):  # x: (B,D)
        return self.fc(x)


# -----------------------
# A11 (noise-only) 的 OVR pairwise logistic（温度 tau）
# -----------------------
def auc_noise_only_a11_loss(
    logits: torch.Tensor,          # (B,C)
    clip_probs: torch.Tensor,      # (B,C)
    s: torch.Tensor,               # (B,)  0/1
    alpha: float = 0.5,            # q = alpha * p_model + (1-alpha) * p_clip
    tau: float = 0.5,              # 温度
    eps: float = 1e-12
) -> torch.Tensor:
    device = logits.device
    C = logits.size(1)

    idx = torch.nonzero(s == 1, as_tuple=False).squeeze(1)
    m = idx.numel()
    if m < 2:
        return logits.new_tensor(0.0, requires_grad=True)

    lo = logits[idx]                         # (m,C)
    p_clip = clip_probs[idx].detach()        # (m,C)
    p_model = F.softmax(lo, dim=-1)          # (m,C)
    q = alpha * p_model + (1.0 - alpha) * p_clip  # (m,C)

    offdiag = ~torch.eye(m, dtype=torch.bool, device=device)
    total_loss = lo.new_tensor(0.0)
    total_w = lo.new_tensor(0.0)

    for i in range(C):
        s_i = lo[:, i]                    # (m,)
        q_i = q[:, i]                     # (m,)
        diff = s_i[:, None] - s_i[None, :]                # (m,m)
        w_posneg = (q_i[:, None]) * (1.0 - q_i)[None, :]  # (m,m)
        w_negpos = (1.0 - q_i[:, None]) * (q_i)[None, :]  # (m,m)

        ell_posneg = F.softplus(-diff / tau)
        ell_negpos = F.softplus( diff / tau)

        w_sum = (w_posneg + w_negpos)
        loss_mat = w_posneg * ell_posneg + w_negpos * ell_negpos

        total_loss = total_loss + (loss_mat * offdiag).sum()
        total_w = total_w + (w_sum * offdiag).sum()

    return total_loss / (total_w + eps)


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

    def step(self, val_loss: float, val_acc: float) -> bool:
        cur = val_loss if self.metric == "loss" else val_acc
        improved = (cur < self.best - self.min_delta) if self.metric == "loss" else (cur > self.best + self.min_delta)
        if improved:
            self.best = cur
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            print(f"早停已用耐心:{self.num_bad}")
            return self.num_bad >= self.patience


# -----------------------
# 记录/绘图小工具
# -----------------------
def _save_val_curves(epochs, losses, accs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # val_acc（%）
    plt.figure()
    plt.plot(epochs, [a * 100.0 for a in accs], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val Acc (%)")
    plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_acc.png"), dpi=150)
    plt.close()

    # val_loss
    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val CE Loss")
    plt.title("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_loss.png"), dpi=150)
    plt.close()


# -----------------------
# 训练主流程
# -----------------------
def train(
    cache_path: str = "./cache/CIFAR100/clip_ViTL14.pt",
    P: int = 16,
    K: int = 16,                 # batch_size = P*K
    lr: float = 1e-3,
    lr_max: float = 5e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    top_outlier_pct: float = 0.30,
    # —— 热身/蒸馏 & AUC 控制
    burn_in_epochs: int = 8,         # 热身周期（仅 CE + 软CE）
    alpha_max: float = 0.5,          # 噪声侧混合权 alpha 上限（用于 AUC 分支）
    alpha_warmup_epochs: int = 20,   # alpha 线性爬升周期
    tau: float = 0.5,                # AUC 温度
    w_auc: float = 0.2,              # AUC 分支目标权重上限
    auc_warmup_epochs: int = 15,     # AUC 权重从 0 -> w_auc 的爬升周期
    # —— 可信 CE 权重策略
    w_ce_base: float = 1.0,
    balance_mode: str = "sqrt",      # ["static", "sqrt", "linear"]
    # —— 监控与保存
    print_every: int = 10,
    ckpt_path: str = "./outputs/CIFAR100/head_auc_ce_new.pt",
    best_ckpt_path: str = "./outputs/CIFAR100/head_auc_ce_new_best.pt",
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

    clip_probs_train = cache["clip_probs_train"].to(torch.float32)
    clip_probs_val   = cache["clip_probs_val"].to(torch.float32)  # 仅占位，未使用
    clip_argmax_train = cache["clip_argmax_train"].long()

    # 单位化特征（健壮）
    train_feats = F.normalize(train_feats, dim=-1)
    val_feats   = F.normalize(val_feats, dim=-1)

    N, D = train_feats.shape
    C = int(max(train_labels.max().item(), val_labels.max().item())) + 1
    print(f"[INFO] Ntrain={N}, Nval={val_labels.numel()}, D={D}, C={C} | P={P}, K={K}, batch={P*K}")

    # 1) 30% 离群人工校正 -> y_obs, s
    y_obs_after, s_flags = outlier_correction_by_class_centers(
        feats=train_feats, y_true=train_labels, y_pseudo=clip_argmax_train, top_pct=top_outlier_pct
    )

    # 2) 数据集/采样器（用 y_obs 做均衡）
    train_ds = FeatureDataset(train_feats, train_labels, y_obs_after, s_flags, clip_probs_train)
    batch_sampler = PKBatchSampler(labels_obs=y_obs_after, P=P, K=K, seed=42)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    # 3) 线性头
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
    val_save_dir = "./val/CIFAR100"
    os.makedirs(val_save_dir, exist_ok=True)
    csv_path = os.path.join(val_save_dir, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_loss", "val_acc"])  # val_acc 为 0~1
    val_epochs, val_losses, val_accs = [], [], []

    print("[TRAIN] Start training ...")

    for epoch in range(1, max_epochs + 1):
        head.train()
        t0 = time.time()

        # alpha 融合 warmup（用于 s=1 的 q，仅正常阶段会被使用）
        lam_alpha = min(1.0, (epoch - 1) / float(alpha_warmup_epochs)) if alpha_warmup_epochs > 0 else 1.0
        alpha = alpha_max * lam_alpha

        running_loss = 0.0
        seen = 0

        for ib, batch in enumerate(train_loader, start=1):
            f, y_true, y_obs, s, p_clip = batch
            f = f.to(device, dtype=torch.float32)
            y_true = y_true.to(device)
            y_obs = y_obs.to(device)
            s = s.to(device)
            p_clip = p_clip.to(device, dtype=torch.float32)

            logits = head(f)
            B = f.size(0)

            # 可信 CE 的有效权重
            mask_t = (s == 0)
            if mask_t.any():
                ce_trusted = ce_criterion(logits[mask_t], y_true[mask_t])
                B_t = int(mask_t.sum().item())
                if balance_mode == "sqrt":
                    w_ce_eff = w_ce_base * math.sqrt(B / max(1, B_t))
                elif balance_mode == "linear":
                    w_ce_eff = w_ce_base * (B / max(1, B_t))
                    w_ce_eff = float(np.clip(w_ce_eff, 0.5, 4.0))
                else:
                    w_ce_eff = w_ce_base
            else:
                ce_trusted = logits.sum() * 0.0
                w_ce_eff = 0.0

            # 阶段一：热身（仅 CE；不可信做“软CE蒸馏”到 CLIP）
            if epoch <= burn_in_epochs:
                stage = "warmup"
                mask_u = (s == 1)
                soft_ce_untrusted = logits.sum() * 0.0
                if mask_u.any():
                    logp = F.log_softmax(logits[mask_u], dim=-1)
                    soft_ce_untrusted = -(p_clip[mask_u] * logp).sum(dim=-1).mean()

                auc_loss = logits.new_tensor(0.0)
                w_auc_eff = 0.0
                loss = w_ce_eff * ce_trusted + soft_ce_untrusted

            # 阶段二：正常（AUC + 可信CE），AUC 权重慢启
            else:
                stage = "main"
                lam_auc = min(1.0, (epoch - burn_in_epochs) / float(max(1, auc_warmup_epochs)))
                w_auc_eff = w_auc * lam_auc

                auc_loss = auc_noise_only_a11_loss(
                    logits=logits,
                    clip_probs=p_clip,
                    s=s,
                    alpha=alpha,
                    tau=tau
                )
                loss = w_auc_eff * auc_loss + w_ce_eff * ce_trusted

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            seen += B

            if ib % print_every == 0:
                contrib_auc = float(auc_loss.item()) * (w_auc_eff if stage == "main" else 0.0)
                contrib_ce  = float(ce_trusted.item()) * w_ce_eff
                print(
                    f"[Epoch {epoch:03d}] batch {ib:04d}/{len(train_loader):04d} | "
                    f"stage={stage} | seen={seen:6d} | alpha={alpha:.2f} | tau={tau:.2f} | "
                    f"w_auc_eff={w_auc_eff:.3f} | w_ce_eff={w_ce_eff:.3f} | "
                    f"contrib_auc={contrib_auc:.5f} | contrib_ce={contrib_ce:.5f} | "
                    f"loss(avg so far)={running_loss/ib:.5f}"
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
        _save_val_curves(val_epochs, val_losses, val_accs, val_save_dir)
        print(f"[VAL-SAVE] CSV -> {csv_path} | PNG -> {val_save_dir}/val_acc.png, val_loss.png")

        # 保存模型
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(head.state_dict(), ckpt_path)
        if best_ckpt_path:
            # 保存“最佳”
            if (early_metric == "loss" and val_loss <= getattr(stopper, "best", float('inf'))) or \
               (early_metric == "acc"  and val_acc  >= getattr(stopper, "best", -float('inf'))):
                torch.save(head.state_dict(), best_ckpt_path)
                print(f"[CKPT] >>> Best so far ({early_metric}) saved to {best_ckpt_path}")

        if stopper.step(val_loss, val_acc):
            print(f"[EARLY STOP] No improvement in {early_metric} for {stopper.patience} epochs. Stop.")
            break

    print("[DONE] Training finished.")


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
    p.add_argument("--top_outlier_pct", type=float, default=0.30)

    # 热身 & AUC
    p.add_argument("--burn_in_epochs", type=int, default=8)
    p.add_argument("--alpha_max", type=float, default=0.5)
    p.add_argument("--alpha_warmup_epochs", type=int, default=20)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--w_auc", type=float, default=0.2)
    p.add_argument("--auc_warmup_epochs", type=int, default=15)

    # 可信 CE 权重
    p.add_argument("--w_ce_base", type=float, default=1.0)
    p.add_argument("--balance_mode", type=str, default="sqrt", choices=["static", "sqrt", "linear"])

    # 日志/保存/早停
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--ckpt", type=str, default="./outputs/CIFAR100/head_auc_ce_new.pt")
    p.add_argument("--best_ckpt", type=str, default="./outputs/CIFAR100/head_auc_ce_new_best.pt")
    p.add_argument("--early_metric", type=str, default="acc", choices=["loss", "acc"])
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(
        cache_path=args.cache,
        P=args.P, K=args.K,
        lr=args.lr, weight_decay=args.weight_decay,
        lr_max=args.lr_max,
        max_epochs=args.max_epochs,
        top_outlier_pct=args.top_outlier_pct,
        burn_in_epochs=args.burn_in_epochs,
        alpha_max=args.alpha_max,
        alpha_warmup_epochs=args.alpha_warmup_epochs,
        tau=args.tau,
        w_auc=args.w_auc,
        auc_warmup_epochs=args.auc_warmup_epochs,
        w_ce_base=args.w_ce_base,
        balance_mode=args.balance_mode,
        print_every=args.print_every,
        ckpt_path=args.ckpt,
        best_ckpt_path=args.best_ckpt,
        early_metric=args.early_metric,
        patience=args.patience,
        min_delta=args.min_delta,
    )
