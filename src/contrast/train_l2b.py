# -*- coding: utf-8 -*-
"""
train_l2b.py

- 读取 extract_CLIP_cifar100.py 生成的严格 45k/5k/10k 缓存（CLIP ViT-L/14 特征已冻结）
- 仅训练线性分类头（FC），在“全部训练样本”上采用 L2B（Learning to Bootstrap）目标：
    L = Σ_i [ α_i * CE(x_i, y_obs_i) + β_i * CE(x_i, y_pseudo_i) ]
  其中 (α_i, β_i) 由“元学习”在每个训练 step 上、基于 5k 验证集（meta-set）进行一步近似更新：
    1) 在当前 batch 上以 (α,β) 做一次“虚步”更新得到 θ_hat；
    2) 在 meta-batch（来自 5k 验证集）上最小化 CE(θ_hat)，对 (α,β) 做一次梯度下降；
    3) 对更新后的 (α,β) 做非负投影与批内归一化；
    4) 用 (α̃,β̃) 回到训练 batch 做真实参数更新。
- 复用既有 warm-up：前 burn_in_epochs 仅用 CE(x, y_obs)，不启用 L2B 的伪标签与元学习；
- 记录每轮的 val_loss / val_acc 到 --val_dir，并保存曲线；
- 早停（按 val_acc 或 val_loss）。
"""

import os, csv, time, argparse
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import clip_cache, observed_labels_cache, experiment_checkpoint, experiment_logs, save_experiment_config

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default=str(clip_cache("CIFAR100")))
    p.add_argument("--obs_labels_path", type=str, default=str(observed_labels_cache("CIFAR100")))
    p.add_argument("--P", type=int, default=16)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--burn_in_epochs", type=int, default=8)
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--meta_lr", type=float, default=0.1)
    p.add_argument("--meta_batch_size", type=int, default=256)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--print_every", type=int, default=20)

    p.add_argument("--best_ckpt", type=str, default=str(experiment_checkpoint("CIFAR100_l2b")))
    p.add_argument("--val_dir", type=str, default=str(experiment_logs("CIFAR100_l2b")))
    p.add_argument("--early_metric", type=str, default="acc", choices=["loss", "acc"])
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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

    needed = [
        "train_feats", "train_labels",
        "val_feats", "val_labels",
        "test_feats", "test_labels",
        # 下面这些即便不用，也尝试兼容读取；若不存在则创建占位
        "clip_probs_train", "clip_probs_val",
        "clip_argmax_train",
    ]
    for k in needed:
        if k not in data:
            print(f"[WARN] {k} not in cache; creating placeholder.")
            if k.startswith("clip_probs"):
                split = "train" if "train" in k else "val"
                feats = data[f"{split}_feats"]
                C = int(max(data["train_labels"].max().item(),
                            data["val_labels"].max().item(),
                            data["test_labels"].max().item()) + 1)
                data[k] = torch.full((feats.size(0), C), 1.0 / C, dtype=torch.float32)
            elif k == "clip_argmax_train":
                C = int(max(data["train_labels"].max().item(),
                            data["val_labels"].max().item(),
                            data["test_labels"].max().item()) + 1)
                data[k] = torch.zeros((data["train_feats"].size(0),), dtype=torch.long)
    return data


# -----------------------
# 读取 make_obs_labels.py 的结果（y_obs, s）
# -----------------------
def load_obs_labels(obs_path: str) -> Dict[str, torch.Tensor]:
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
# 特征数据集 & P×K 均衡采样（类别按 y_obs 均衡）
# -----------------------
class FeatureDataset(Dataset):
    def __init__(self, feats, y_true, y_obs, s, p_clip=None):
        self.f = feats
        self.y_true = y_true
        self.y_obs = y_obs
        self.s = s
        self.p_clip = p_clip if p_clip is not None else torch.empty((feats.size(0), 1), dtype=torch.float32)
    def __len__(self): return self.f.size(0)
    def __getitem__(self, idx):
        return self.f[idx], int(self.y_true[idx]), int(self.y_obs[idx]), int(self.s[idx]), self.p_clip[idx]


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
        self.steps_per_epoch = len(self.labels) // max(1, (self.P * self.K))

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
            if len(self.classes) >= self.P:
                chosen = rng.sample(self.classes, self.P)
            else:
                chosen = [rng.choice(self.classes) for _ in range(self.P)]
            batch_idx = []
            for c in chosen:
                batch_idx.extend(self._next_k(c, self.K))
            rng.shuffle(batch_idx)
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
# 评估 & 早停 & 曲线
# -----------------------
def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048) -> Tuple[float, float]:
    head.eval()
    ds = TensorDataset(feats, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ce = nn.CrossEntropyLoss(reduction="mean")
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in dl:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device)
        with torch.no_grad():
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


def _save_val_curves(epochs, losses, accs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # val_acc
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
# L2B-Linear 一步元学习实现（不依赖 higher）
# -----------------------
def l2b_meta_update_and_loss(
    head: nn.Module,
    f_train: torch.Tensor, y_obs_train: torch.Tensor,
    f_meta: torch.Tensor, y_meta: torch.Tensor,
    inner_lr: float = 1e-3, meta_lr: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    输入：
      head: 线性头
      f_train, y_obs_train: 训练 batch
      f_meta, y_meta:       元（验证）batch
    输出：
      final_loss: 用 (α̃,β̃) 在训练 batch 上计算的真实更新损失（标量）
      alpha_tilde_mean, beta_tilde_mean: 当前 batch 的均值，用于监控
    """
    head.train()

    # 1) 前向，得到 per-sample CE_real / CE_pseudo（向量）
    logits = head(f_train)
    ce_vec_real = F.cross_entropy(logits, y_obs_train, reduction="none")  # (B,)
    with torch.no_grad():
        y_pseudo = logits.argmax(dim=-1)
    ce_vec_pseudo = F.cross_entropy(logits, y_pseudo, reduction="none")   # (B,)

    B = f_train.size(0)
    # α,β 为“当前 batch 的变量”，不跨 batch/epoch 持久化
    alpha = torch.ones(B, device=f_train.device, requires_grad=True)
    beta  = torch.zeros(B, device=f_train.device, requires_grad=True)

    # 2) “虚步”：以 (α,β) 对 θ 做一次内层更新，得到 θ_hat
    W = head.fc.weight        # (C,D)
    b = head.fc.bias          # (C,)
    loss_train = (alpha * ce_vec_real + beta * ce_vec_pseudo).mean()
    grad_W, grad_b = torch.autograd.grad(loss_train, [W, b], create_graph=True)

    W_hat = W - inner_lr * grad_W
    b_hat = b - inner_lr * grad_b

    # 3) 外层 meta-loss：在验证 batch 上，用 θ_hat 计算 CE，并 w.r.t. (α,β) 做一次下降
    logits_meta = F.linear(f_meta, W_hat, b_hat)
    loss_meta = F.cross_entropy(logits_meta, y_meta, reduction="mean")

    grad_alpha, grad_beta = torch.autograd.grad(loss_meta, (alpha, beta), allow_unused=True)
    if grad_alpha is None: grad_alpha = torch.zeros_like(alpha)
    if grad_beta  is None: grad_beta  = torch.zeros_like(beta)

    # 一步梯度更新 + 非负投影
    alpha_tilde = torch.clamp(alpha - meta_lr * grad_alpha, min=0.0)
    beta_tilde  = torch.clamp(beta  - meta_lr * grad_beta,  min=0.0)

    # 批内归一化：让 mean(α̃+β̃)=1
    denom = (alpha_tilde + beta_tilde).mean().detach().clamp(min=1e-8)
    alpha_tilde = alpha_tilde / denom
    beta_tilde  = beta_tilde  / denom

    # 用 (α̃,β̃) 计算最终训练损失（切断与 meta 图的梯度）
    final_loss = (alpha_tilde.detach() * ce_vec_real + beta_tilde.detach() * ce_vec_pseudo).mean()

    return final_loss, alpha_tilde.mean().detach(), beta_tilde.mean().detach()


# -----------------------
# 训练
# -----------------------
def train(
    cache_path: str = str(clip_cache("CIFAR100")),
    obs_labels_path: str = str(observed_labels_cache("CIFAR100")),
    # 采样与优化
    P: int = 16, K: int = 16,                 # batch_size = P*K
    max_epochs: int = 200,
    lr: float = 1e-3, weight_decay: float = 1e-4,
    # warm-up（前若干 epoch 仅 CE，不启用 L2B）
    burn_in_epochs: int = 8,
    # L2B 超参
    inner_lr: float = 1e-3, meta_lr: float = 0.1,
    meta_batch_size: int = 256,
    grad_clip: float = 1.0,
    # 日志/模型/早停
    print_every: int = 50,
    best_ckpt_path: str = str(experiment_checkpoint("CIFAR100_l2b")),
    val_save_dir: str = str(experiment_logs("CIFAR100_l2b")),
    early_metric: str = "acc",
    patience: int = 5,
    min_delta: float = 0.0,
    seed: int = 42,
):
    set_seed(seed)

    # 0) 载入缓存并单位化特征
    cache = load_clip_cache_strict(cache_path)
    train_feats = cache["train_feats"].to(torch.float32)
    val_feats   = cache["val_feats"].to(torch.float32)
    train_labels = cache["train_labels"].long()
    val_labels   = cache["val_labels"].long()

    train_feats = F.normalize(train_feats, dim=-1)
    val_feats   = F.normalize(val_feats, dim=-1)

    N, D = train_feats.shape
    C = int(max(train_labels.max().item(), val_labels.max().item())) + 1
    print(f"[INFO] Ntrain={N}, Nval={val_labels.numel()}, D={D}, C={C} | P={P}, K={K}, batch={P*K}")

    # 1) 读取 make_obs_labels.py 的输出（y_obs, s），替代脚本内“离群人工校正”
    obs = load_obs_labels(obs_labels_path)
    y_obs, s_mark = obs["y_obs"], obs["s"]
    if y_obs.numel() != N or s_mark.numel() != N:
        raise RuntimeError(f"Obs length mismatch: feats N={N}, y_obs={y_obs.numel()}, s={s_mark.numel()}")

    # 2) 数据集 & 采样器（按 y_obs 做类均衡）
    ds_train = FeatureDataset(train_feats, train_labels, y_obs, s_mark, p_clip=None)
    batch_sampler = PKBatchSampler(labels_obs=y_obs, P=P, K=K, seed=seed)
    train_loader = DataLoader(ds_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    # meta loader（从 5k 验证集中随机采样 meta_batch）
    ds_meta = TensorDataset(val_feats, val_labels)
    meta_loader = DataLoader(ds_meta, batch_size=meta_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    meta_iter = iter(meta_loader)

    # 3) 头部与优化器
    head = LinearHead(D, C).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    # 4) 监控与早停
    os.makedirs(val_save_dir, exist_ok=True)
    csv_path = os.path.join(val_save_dir, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_loss", "val_acc", "alpha_mean", "beta_mean"])  # val_acc 为 0~1
    val_epochs, val_losses, val_accs = [], [], []
    stopper = EarlyStopping(metric=early_metric, patience=patience, min_delta=min_delta)

    print("[TRAIN] Start L2B-Linear training ...")

    alpha_running, beta_running = 1.0, 0.0

    for epoch in range(1, max_epochs + 1):
        head.train()
        t0 = time.time()

        running_loss = 0.0
        seen = 0
        alpha_epoch_sum, beta_epoch_sum, batches = 0.0, 0.0, 0

        for ib, batch in enumerate(train_loader, start=1):
            f, y_true, y_obs_b, s_b, _ = batch
            f = f.to(device, dtype=torch.float32)
            y_obs_b = y_obs_b.to(device)

            # 取 meta-batch；若迭代器耗尽则重建
            try:
                f_meta, y_meta = next(meta_iter)
            except StopIteration:
                meta_iter = iter(meta_loader)
                f_meta, y_meta = next(meta_iter)
            f_meta = f_meta.to(device, dtype=torch.float32)
            y_meta = y_meta.to(device)

            optimizer.zero_grad(set_to_none=True)

            # warm-up：仅 CE(y_obs)
            if epoch <= burn_in_epochs:
                logits = head(f)
                loss = F.cross_entropy(logits, y_obs_b, reduction="mean")
                alpha_mean = torch.tensor(1.0, device=f.device)
                beta_mean  = torch.tensor(0.0, device=f.device)
            else:
                # L2B 一步元学习
                loss, alpha_mean, beta_mean = l2b_meta_update_and_loss(
                    head, f, y_obs_b, f_meta, y_meta, inner_lr=inner_lr, meta_lr=meta_lr
                )

            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_loss += float(loss.item())
            seen += f.size(0)
            alpha_epoch_sum += float(alpha_mean.item())
            beta_epoch_sum  += float(beta_mean.item())
            batches += 1

            if ib % print_every == 0:
                print(f"[Epoch {epoch:03d}] batch {ib:04d}/{len(train_loader):04d} | "
                      f"seen={seen:6d} | loss(avg)={running_loss/batches:.5f} | "
                      f"alpha≈{alpha_epoch_sum/batches:.3f} | beta≈{beta_epoch_sum/batches:.3f}")

        dt = time.time() - t0
        alpha_running = alpha_epoch_sum / max(1, batches)
        beta_running  = beta_epoch_sum  / max(1, batches)
        print(f"[Epoch {epoch:03d}] time={dt:.1f}s | train_loss={running_loss/max(1,batches):.5f} "
              f"| alpha≈{alpha_running:.3f} | beta≈{beta_running:.3f}")

        # 验证与早停
        val_loss, val_acc = evaluate(head, val_feats, val_labels, batch_size=2048)
        print(f"[VAL] epoch={epoch:03d} | val_loss={val_loss:.5f} | val_acc={val_acc*100:.2f}%")

        # 记录、写入 CSV、刷新曲线
        val_epochs.append(epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.6f}", f"{alpha_running:.6f}", f"{beta_running:.6f}"])
        _save_val_curves(val_epochs, val_losses, val_accs, val_save_dir)
        print(f"[VAL-SAVE] CSV -> {csv_path} | PNG -> {val_save_dir}/val_acc.png, val_loss.png")

        # 仅保存“最佳”
        if (early_metric == "loss" and val_loss <= getattr(stopper, "best", float('inf'))) or \
           (early_metric == "acc"  and val_acc  >= getattr(stopper, "best", -float('inf'))):
            if best_ckpt_path:
                os.makedirs(os.path.dirname(best_ckpt_path) or ".", exist_ok=True)
                torch.save(head.state_dict(), best_ckpt_path)
                print(f"[CKPT] >>> Best so far ({early_metric}) saved to {best_ckpt_path}")

        # 早停
        if stopper.step(val_loss, val_acc):
            print(f"[EARLY STOP] Stop at epoch={epoch}. Best {early_metric}={stopper.best:.6f}.")
            break

    print("[DONE] L2B-Linear training finished.")


# -----------------------
# Argparse
# -----------------------
if __name__ == "__main__":
    args = parse_args()
    save_experiment_config("CIFAR100_l2b", args)
    set_seed(args.seed)
    train(
        cache_path=args.cache,
        obs_labels_path=args.obs_labels_path,
        P=args.P, K=args.K,
        max_epochs=args.max_epochs,
        lr=args.lr, weight_decay=args.weight_decay,
        burn_in_epochs=args.burn_in_epochs,
        inner_lr=args.inner_lr, meta_lr=args.meta_lr, meta_batch_size=args.meta_batch_size,
        grad_clip=args.grad_clip,
        print_every=args.print_every,
        best_ckpt_path=args.best_ckpt, val_save_dir=args.val_dir,
        early_metric=args.early_metric, patience=args.patience, min_delta=args.min_delta,
        seed=args.seed,
    )
