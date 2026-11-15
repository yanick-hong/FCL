# -*- coding: utf-8 -*-
"""
train_dld_CIFAR100.py

使用示例：
python train_dld_CIFAR100.py \
  --cache ./cache/CIFAR100/clip_ViTL14.pt \
  --P 32 --K 32 --max_epochs 200 --top_outlier_pct 0.30 \
  --steps 5 --tau_clip 0.7 --hidden 512

可在训练完成后运行：
python train_dld_CIFAR100.py --cache ./cache/CIFAR100/clip_ViTL14.pt --eval-only --ckpt ./outputs/CIFAR100/head_dld_best.pt --split test
"""

import os
import math
import csv
import time
import argparse
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# 通用 & 工具
# -----------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clip_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
    """仅读取严格 45k/5k/10k 缓存（由 extract_CLIP_cifar100.py 生成）。"""
    assert os.path.isfile(cache_path), f"Cache not found: {cache_path}"
    print(f"[CACHE] Loading from: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

    tensor_fp32_keys = [
        "train_feats", "val_feats", "test_feats",
        "clip_probs_train", "clip_probs_val", "clip_probs_test",
        "clip_label_embeds",
    ]
    tensor_long_keys = [
        "train_labels", "val_labels", "test_labels",
        "clip_argmax_train", "clip_argmax_val", "clip_argmax_test",
        "train_idx_from_train50k", "val_idx_from_train50k",
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
    s     = torch.ones(N, dtype=torch.long)  # s 未用于 DLD，仅为兼容输出
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
        # 人工校正：用真值替换
        y_obs[outlier_idx_global] = y_true[outlier_idx_global]
        s[outlier_idx_global] = 0
        if c % 10 == 0:
            print(f"[STEP 1] class={c:03d} | size={len(idx):4d} | corrected={k:4d}")
    total_trusted = int((s == 0).sum().item())
    print(f"[STEP 1] Trusted(s=0): {total_trusted} | Others: {N - total_trusted}")
    return y_obs, s


# -----------------------
# 特征数据集 & P×K 采样
# -----------------------

class FeatureDataset(Dataset):
    """返回：(f, y_true, y_obs, p_clip)。DLD 仅用 y_obs 和 p_clip。"""
    def __init__(self, feats, y_true, y_obs, p_clip):
        self.f = feats
        self.y_true = y_true
        self.y_obs = y_obs
        self.p_clip = p_clip
    def __len__(self):
        return self.f.size(0)
    def __getitem__(self, idx):
        return self.f[idx], int(self.y_true[idx]), int(self.y_obs[idx]), self.p_clip[idx]


class PKBatchSampler(Sampler[List[int]]):
    """每 batch 采样 P 个类别、每类 K 个样本 -> batch_size = P*K。用 y_obs 做均衡。"""
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
# DLD-Linear 模型
# -----------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):  # t in [0,1], shape (B,)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=device)
        )  # [1, 1000] 对数均匀
        angles = t[:, None] * freqs[None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1), value=0.0)
        return emb  # (B, dim)


class DLDLinear(nn.Module):
    """
    输入：x(=f), y_t, y_n, t
    输出：y_d_hat, eps_hat  （均为 C 维）
    """
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, tdim: int = 64):
        super().__init__()
        C = num_classes
        H = hidden
        self.t_embed = SinusoidalTimeEmbedding(tdim)
        self.proj_f = nn.Sequential(
            nn.Linear(in_dim, H), nn.GELU(), nn.LayerNorm(H)
        )
        self.proj_y = nn.Sequential(
            nn.Linear(2 * C, H), nn.GELU(), nn.LayerNorm(H)
        )
        self.proj_t = nn.Sequential(
            nn.Linear(tdim, H), nn.GELU(), nn.LayerNorm(H)
        )
        self.head = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H)
        )
        self.out_d   = nn.Linear(H, C)
        self.out_eps = nn.Linear(H, C)
        # 初始化
        for m in [self.out_d, self.out_eps]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y_t, y_n, t):
        # x:(B,D)  y_t/y_n:(B,C)  t:(B,)
        t_emb = self.t_embed(t)
        h = self.proj_f(x) + self.proj_y(torch.cat([y_t, y_n], dim=-1)) + self.proj_t(t_emb)
        h = self.head(h)
        y_d_hat   = self.out_d(h)
        eps_hat   = self.out_eps(h)
        return y_d_hat, eps_hat


# -----------------------
# DLD 前向/反向调度 & 工具
# -----------------------

def a_func(t: torch.Tensor) -> torch.Tensor:
    """a(t) in [0,1]，线性调度即可。"""
    return t

def b_func(t: torch.Tensor) -> torch.Tensor:
    """b(t) in [0,1]，线性调度（可改为 sqrt(t)）。"""
    return t


def one_hot(y: torch.Tensor, C: int) -> torch.Tensor:
    return F.one_hot(y.clamp_min(0), num_classes=C).to(torch.float32)


def smooth_probs(p: torch.Tensor, tau: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    """对 CLIP 概率做温度平滑与截断，避免极端 0。"""
    p = p.float().clamp_min(eps)
    logp = torch.log(p)
    logp = logp / max(1e-6, tau)
    q = F.softmax(logp, dim=-1)
    return q


@torch.no_grad()
def dld_predict_proba(model: DLDLinear, feats: torch.Tensor, p_clip: torch.Tensor,
                      steps: int = 5, batch_size: int = 2048) -> torch.Tensor:
    """对给定样本做 DLD 反向（DDIM-like）采样，返回 softmax(y0_hat)。
    - y_n 取平滑后的 p_clip；
    - 初始 y_T ← y_n（不加随机噪声，走确定性路径）；
    """
    model.eval()
    N, C = p_clip.size(0), p_clip.size(1)
    probs_out = []

    dl = DataLoader(TensorDataset(feats, p_clip), batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=(device.type == "cuda"))
    t_grid = torch.linspace(1.0, 0.0, steps=steps + 1, device=device)  # t_K=1 -> t_0=0

    for xb, pb in dl:
        xb = xb.to(device).float()
        pb = pb.to(device).float()
        y_n = smooth_probs(pb)
        y_t = y_n.clone()  # 以 y_n 为起点
        # 反向迭代
        for k in range(steps, 0, -1):
            t_k = torch.full((xb.size(0),), float(t_grid[k].item()), device=device)
            t_km1 = torch.full((xb.size(0),), float(t_grid[k-1].item()), device=device)
            a_k = a_func(t_k)
            b_k = b_func(t_k)
            y_d_hat, eps_hat = model(xb, y_t, y_n, t_k)
            # 估计 y0
            y0_hat = y_t - a_k[:, None] * y_d_hat - b_k[:, None] * eps_hat
            # DDIM：下一步 y_{t_{k-1}}
            a_km1 = a_func(t_km1)
            # 这里取确定性路径，不加噪声项
            y_t = y0_hat + a_km1[:, None] * y_d_hat
        probs = F.softmax(y0_hat, dim=-1)
        probs_out.append(probs.detach().cpu())

    return torch.cat(probs_out, dim=0)  # (N,C)


# -----------------------
# 评估 & 早停 & 记录
# -----------------------

@torch.no_grad()
def evaluate_dld(model: DLDLinear, feats: torch.Tensor, labels: torch.Tensor,
                 p_clip: torch.Tensor, steps: int = 5, batch_size: int = 2048) -> Tuple[float, float]:
    # 以 DLD 采样得到概率，再计算 CE 与 Top-1
    probs = dld_predict_proba(model, feats, p_clip, steps=steps, batch_size=batch_size).to(device)
    labels = labels.to(device)
    ce = F.nll_loss((probs + 1e-12).log(), labels, reduction="mean")
    pred = probs.argmax(dim=-1)
    acc = (pred == labels).float().mean().item()
    return float(ce.item()), float(acc)


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


def _save_val_curves(epochs, losses, accs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(epochs, [a * 100.0 for a in accs], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Val Acc (%)"); plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_acc.png"), dpi=150); plt.close()

    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Val CE Loss"); plt.title("Validation Loss")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_loss.png"), dpi=150); plt.close()


# -----------------------
# 训练主流程（DLD-Linear）
# -----------------------

def train(
    cache_path: str = "./cache/CIFAR100/clip_ViTL14.pt",
    P: int = 16,
    K: int = 16,
    lr: float = 1e-3,
    lr_max: float = 5e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    top_outlier_pct: float = 0.30,
    # DLD 配置
    hidden: int = 512,
    tdim: int = 64,
    steps: int = 5,
    tau_clip: float = 0.7,
    # 训练细节
    print_every: int = 20,
    ckpt_path: str = "./outputs/CIFAR100/head_dld.pt",
    best_ckpt_path: str = "./outputs/CIFAR100/head_dld_best.pt",
    early_metric: str = "acc",
    patience: int = 10,
    min_delta: float = 0.0,
):
    set_seed(42)

    # 0) 载入缓存
    cache = load_clip_cache_strict(cache_path)
    train_feats = F.normalize(cache["train_feats"].to(torch.float32), dim=-1)
    val_feats   = F.normalize(cache["val_feats"].to(torch.float32), dim=-1)
    train_labels = cache["train_labels"].long()
    val_labels   = cache["val_labels"].long()
    clip_probs_train = cache["clip_probs_train"].to(torch.float32)
    clip_probs_val   = cache.get("clip_probs_val", None)
    if clip_probs_val is None:
        raise RuntimeError("缓存中缺少 clip_probs_val ，无法进行 DLD 验证（需要 y_n）。")
    clip_probs_val = clip_probs_val.to(torch.float32)

    N, D = train_feats.shape
    C = int(max(train_labels.max().item(), val_labels.max().item())) + 1
    print(f"[INFO] Ntrain={N}, Nval={val_labels.numel()}, D={D}, C={C} | P={P}, K={K}, batch={P*K}")

    # 1) 离群 30% 人工校正 -> y_obs（和主实验一致），但 DLD 不再区分 s
    y_obs_after, _ = outlier_correction_by_class_centers(
        feats=train_feats, y_true=train_labels, y_pseudo=cache["clip_argmax_train"].long(), top_pct=top_outlier_pct
    )

    # 2) 数据集与均衡采样（按 y_obs）
    train_ds = FeatureDataset(train_feats, train_labels, y_obs_after, clip_probs_train)
    batch_sampler = PKBatchSampler(labels_obs=y_obs_after, P=P, K=K, seed=42)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    # 3) DLD-Linear 模型
    model = DLDLinear(in_dim=D, num_classes=C, hidden=hidden, tdim=tdim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_max,
        total_steps=max(1, steps_per_epoch * max_epochs),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=10.0,
    )

    stopper = EarlyStopping(metric=early_metric, patience=patience, min_delta=min_delta)

    # 记录文件
    val_save_dir = "./val/CIFAR100"
    os.makedirs(val_save_dir, exist_ok=True)
    csv_path = os.path.join(val_save_dir, "val_metrics_dld.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["epoch", "val_loss", "val_acc"])  # acc 为 0~1
    val_epochs, val_losses, val_accs = [], [], []

    print("[TRAIN] Start DLD-Linear training ...")
    mse = nn.MSELoss(reduction="mean")

    global_step = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        seen = 0

        for ib, batch in enumerate(train_loader, start=1):
            f, y_true, y_obs, p_clip = batch
            f = f.to(device, dtype=torch.float32)
            y_obs = y_obs.to(device)
            p_clip = p_clip.to(device, dtype=torch.float32)

            B = f.size(0)
            y0 = one_hot(y_obs, C)                              # (B,C)
            y_n = smooth_probs(p_clip, tau=tau_clip)            # (B,C)
            y_d = (y_n - y0)                                    # (B,C)

            # 采样 t 与 eps，合成 y_t
            t = torch.rand(B, device=device)  # U(0,1)
            a_t = a_func(t)                    # (B,)
            b_t = b_func(t)                    # (B,)
            eps = torch.randn_like(y0)         # (B,C)
            y_t = y0 + a_t[:, None] * y_d + b_t[:, None] * eps

            # 预测与损失
            y_d_hat, eps_hat = model(f, y_t, y_n, t)
            loss = mse(y_d_hat, y_d) + mse(eps_hat, eps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step(); scheduler.step()

            run_loss += float(loss.item())
            seen += B
            global_step += 1

            if ib % print_every == 0:
                print(
                    f"[Epoch {epoch:03d}] batch {ib:04d}/{len(train_loader):04d} | "
                    f"seen={seen:6d} | lr={scheduler.get_last_lr()[0]:.3e} | loss(avg)={run_loss/ib:.5f}"
                )

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] time={dt:.1f}s | avg_loss={run_loss/max(1,len(train_loader)):.5f}")

        # 验证：DLD 反向采样得到概率
        val_loss, val_acc = evaluate_dld(model, val_feats, val_labels, clip_probs_val, steps=steps, batch_size=2048)
        print(f"[VAL] epoch={epoch:03d} | val_loss={val_loss:.5f} | val_acc={val_acc*100:.2f}%")

        # 记录曲线
        val_epochs.append(epoch); val_losses.append(val_loss); val_accs.append(val_acc)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f); writer.writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.6f}"])
        _save_val_curves(val_epochs, val_losses, val_accs, val_save_dir)
        print(f"[VAL-SAVE] CSV -> {csv_path} | PNG -> {val_save_dir}/val_acc.png, val_loss.png")

        # 保存 ckpt（同时保存 meta 便于 eval-only）
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        payload = {
            "model": model.state_dict(),
            "in_dim": D,
            "num_classes": C,
            "hidden": hidden,
            "tdim": tdim,
        }
        torch.save(payload, ckpt_path)

        # best ckpt 按 early_metric 判定
        improved = (early_metric == "loss" and val_loss <= getattr(stopper, "best", float('inf'))) or \
                   (early_metric == "acc"  and val_acc  >= getattr(stopper, "best", -float('inf')))
        if improved and best_ckpt_path:
            torch.save(payload, best_ckpt_path)
            print(f"[CKPT] >>> Best so far ({early_metric}) saved to {best_ckpt_path}")

        if stopper.step(val_loss, val_acc):
            print(f"[EARLY STOP] No improvement in {early_metric} for {stopper.patience} epochs. Stop.")
            break

    print("[DONE] DLD-Linear training finished.")


# -----------------------
# 仅评估模式（train 完成后在 train/val/test 上做 DLD 推断）
# -----------------------

def eval_only(cache_path: str, ckpt: str, split: str = "test", steps: int = 5, batch_size: int = 2048):
    cache = load_clip_cache_strict(cache_path)
    if split == "train":
        feats, labels, p = cache["train_feats"].float(), cache["train_labels"].long(), cache["clip_probs_train"].float()
    elif split == "val":
        feats, labels, p = cache["val_feats"].float(), cache["val_labels"].long(), cache["clip_probs_val"].float()
    elif split == "test":
        feats, labels, p = cache["test_feats"].float(), cache["test_labels"].long(), cache["clip_probs_test"].float()
    else:
        raise ValueError(f"Unknown split: {split}")

    feats = F.normalize(feats, dim=-1)
    in_dim = feats.size(1)
    num_classes = int(max(labels.max().item() + 1, cache.get("clip_label_embeds", torch.empty(0)).shape[0] if "clip_label_embeds" in cache else 100))

    payload = torch.load(ckpt, map_location="cpu")
    model = DLDLinear(in_dim=in_dim, num_classes=num_classes,
                      hidden=int(payload.get("hidden", 512)), tdim=int(payload.get("tdim", 64)))
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()

    ce, acc = evaluate_dld(model, feats, labels, p, steps=steps, batch_size=batch_size)
    print(f"[EVAL-ONLY] split={split} | CE={ce:.4f} | Top1={acc*100:.2f}% | N={labels.numel()}")


# -----------------------
# main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser()
    # 数据与采样
    p.add_argument("--cache", type=str, default="./cache/CIFAR100/clip_ViTL14.pt")
    p.add_argument("--P", type=int, default=32)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--top_outlier_pct", type=float, default=0.30)
    # 优化
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_max", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=200)
    # DLD & 网络
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--tdim", type=int, default=64)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--tau_clip", type=float, default=0.7)
    # 训练/日志
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--ckpt", type=str, default="./outputs/CIFAR100/temp.pt")
    p.add_argument("--best_ckpt", type=str, default="./outputs/CIFAR100/head_dld_best.pt")
    p.add_argument("--early_metric", type=str, default="acc", choices=["loss", "acc"])
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    # 评估仅模式
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=2048)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    if args.eval_only:
        eval_only(cache_path=args.cache, ckpt=args.ckpt, split=args.split, steps=args.steps, batch_size=args.batch_size)
    else:
        train(
            cache_path=args.cache,
            P=args.P, K=args.K,
            lr=args.lr, lr_max=args.lr_max, weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            top_outlier_pct=args.top_outlier_pct,
            hidden=args.hidden, tdim=args.tdim, steps=args.steps, tau_clip=args.tau_clip,
            print_every=args.print_every,
            ckpt_path=args.ckpt, best_ckpt_path=args.best_ckpt,
            early_metric=args.early_metric, patience=args.patience, min_delta=args.min_delta,
        )
