# -*- coding: utf-8 -*-
"""
train_csgn.py
"""

import os, time, math, argparse, csv
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

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
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--cache_path", type=str, default=str(clip_cache("CIFAR100")))
    parser.add_argument("--obs_labels_path", type=str, default=str(observed_labels_cache("CIFAR100")))
    parser.add_argument("--seed", type=int, default=42)

    # 训练
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--P", type=int, default=32)
    parser.add_argument("--K", type=int, default=16)

    # 模型/损失
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--lambda_sup", type=float, default=0.1)
    parser.add_argument("--topk_y", type=int, default=5)
    parser.add_argument("--mc_L", type=int, default=1)

    # 优化
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # 输出
    parser.add_argument("--val_dir", type=str, default=str(experiment_logs("CIFAR100_csgn")),
                        help="验证日志输出目录（CSV/曲线）")
    parser.add_argument("--best_ckpt", type=str, default=str(experiment_checkpoint("CIFAR100_csgn")),
                        help="最佳模型保存路径（整合了原 out_dir 与 tag）")
    return parser.parse_args()

# -----------------------
# 通用
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# -----------------------
# 严格读取缓存
# -----------------------
def load_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

    required = ["train_feats","train_labels","val_feats","val_labels","test_feats","test_labels"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"缓存缺少必需键: {missing}。")

    for k in ["train_feats","val_feats","test_feats","clip_label_embeds","clip_probs_train","clip_probs_val","clip_probs_test"]:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(torch.float32)
    for k in ["train_labels","val_labels","test_labels"]:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(torch.long)
    return data

# -----------------------
# 读取 y_obs（来自 make_obs_labels.py）
# -----------------------
def load_y_obs(obs_labels_path: str, n_train: int):
    if not os.path.exists(obs_labels_path):
        raise FileNotFoundError(f"Obs labels not found: {obs_labels_path}")
    obj = torch.load(obs_labels_path, map_location="cpu")

    y_obs = None; s = None; meta = None
    if isinstance(obj, torch.Tensor):
        y_obs = obj
    elif isinstance(obj, dict):
        if "y_obs" in obj: y_obs = obj["y_obs"]
        elif "obs_labels" in obj: y_obs = obj["obs_labels"]
        if "s" in obj: s = obj["s"]
        if "meta" in obj and isinstance(obj["meta"], dict): meta = obj["meta"]
    else:
        raise RuntimeError(f"{obs_labels_path} 格式不支持。")

    if y_obs is None:
        raise RuntimeError(f"{obs_labels_path} 未找到 y_obs。")
    y_obs = y_obs.to(torch.long).view(-1)
    if y_obs.numel() != n_train:
        raise RuntimeError(f"y_obs 长度 {y_obs.numel()} 与训练集样本数 {n_train} 不符。")

    if s is not None:
        s = s.to(torch.long).view(-1)
        if s.numel() != n_train:
            print(f"[WARN] s 长度 {s.numel()} != 训练集 {n_train}，忽略 s。")
            s = None

    if s is not None:
        num_trusted = int((s==0).sum().item())
        print(f"[OBS] y_obs loaded. trusted s=0: {num_trusted}  |  untrusted s=1: {int(s.numel())-num_trusted}")
    else:
        print(f"[OBS] y_obs loaded. (no s provided)")
    return y_obs, s

# -----------------------
# P×K 均衡采样（按 y_obs）
# -----------------------
class BalancedPKSampler(Sampler[List[int]]):
    def __init__(self, labels: torch.Tensor, P: int, K: int, drop_last: bool = True):
        self.labels = labels.cpu().numpy()
        self.P = int(P); self.K = int(K)
        self.batch_size = self.P * self.K
        self.drop_last = drop_last
        self.num_classes = int(self.labels.max()) + 1
        self.idxs_by_cls = [np.where(self.labels == c)[0].tolist() for c in range(self.num_classes)]
        self.num_samples = len(self.labels)
        self.valid_classes = [c for c in range(self.num_classes) if len(self.idxs_by_cls[c]) > 0]
        if len(self.valid_classes) < self.P:
            raise ValueError(f"可用类别数 {len(self.valid_classes)} < P={self.P}")
        self.batches_per_epoch = (self.num_samples // self.batch_size)

    def __len__(self):
        # This sampler yields complete batches and is passed as ``batch_sampler``.
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.batches_per_epoch):
            classes = rng.choice(self.valid_classes, size=self.P, replace=False)
            batch = []
            for c in classes:
                pool = self.idxs_by_cls[c]
                if len(pool) >= self.K:
                    picks = rng.choice(pool, size=self.K, replace=False)
                else:
                    picks = rng.choice(pool, size=self.K, replace=True)
                batch.extend(picks.tolist())
            rng.shuffle(batch)
            yield batch

# -----------------------
# 数据集
# -----------------------
class FeaturesWithNoisyLabels(Dataset):
    def __init__(self, feats: torch.Tensor, y_obs: torch.Tensor):
        self.x = feats; self.y_obs = y_obs
        assert len(self.x) == len(self.y_obs)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y_obs[idx]

# -----------------------
# CSGN-Linear 组件（特征域）
# -----------------------
class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
    def forward(self, x): return self.fc(x)

class EncoderZX(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, z_dim: int, hidden: int = 256):
        super().__init__()
        self.y_emb = nn.Embedding(num_classes, hidden)
        self.x_proj = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        nn.init.zeros_(self.y_emb.weight)
    def forward(self, x: torch.Tensor, y_cls: torch.Tensor):
        h = self.act(self.x_proj(x) + self.y_emb(y_cls))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-8.0, max=8.0)
        return mu, logvar

class PriorZY(nn.Module):
    def __init__(self, num_classes: int, z_dim: int):
        super().__init__()
        self.mu = nn.Embedding(num_classes, z_dim)
        self.logvar = nn.Embedding(num_classes, z_dim)
        nn.init.zeros_(self.mu.weight)
        nn.init.constant_(self.logvar.weight, -2.0)
    def forward(self, y_cls: torch.Tensor):
        return self.mu(y_cls), self.logvar(y_cls)

class DecoderX(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, logvar_x: float = -2.0):
        super().__init__()
        self.fc = nn.Linear(z_dim, out_dim)
        self.logvar_x = nn.Parameter(torch.tensor([logvar_x], dtype=torch.float32), requires_grad=False)
    def forward(self, z):
        mu_x = self.fc(z)
        return mu_x, self.logvar_x

class DecoderYtilde(nn.Module):
    def __init__(self, z_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, num_classes)
    def forward(self, z): return self.fc(z)

# -----------------------
# ELBO / 损失
# -----------------------
def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * ( (var_q/var_p).sum(dim=-1)
               + ((mu_p - mu_q).pow(2)/var_p).sum(dim=-1)
               - mu_q.size(-1)
               + (logvar_p - logvar_q).sum(dim=-1) )
    return kl

def log_gaussian_likelihood(x, mu, logvar):
    D = x.size(-1)
    if logvar.numel()==1:
        log_det = D * logvar
        inv_var = torch.exp(-logvar)
        sq = ((x - mu).pow(2) * inv_var).sum(dim=-1)
        const = D * math.log(2*math.pi)
        return -0.5 * (const + log_det + sq)
    else:
        raise NotImplementedError("仅实现固定同方差。")

def elbo_batch(x, y_tilde, model, topk: int = 5, L: int = 1, lambda_sup: float = 0.1):
    clf: ClassifierHead = model["clf"]
    enc: EncoderZX = model["enc"]
    prior: PriorZY = model["prior"]
    dec_x: DecoderX = model["dec_x"]
    dec_y: DecoderYtilde = model["dec_y"]

    B, D = x.size()
    C = clf.fc.out_features

    logits_y = clf(x)
    qy = F.softmax(logits_y, dim=-1)
    sup_loss = F.cross_entropy(logits_y, y_tilde)

    topk = min(topk, C)
    qy_topk, y_topk = torch.topk(qy, k=topk, dim=-1)      # (B,K)
    weight = qy_topk / (qy_topk.sum(dim=-1, keepdim=True) + 1e-12)

    y_flat = y_topk.reshape(-1)
    x_rep = x.unsqueeze(1).expand(B, topk, D).reshape(B*topk, D)

    mu_q, logvar_q = enc(x_rep, y_flat)
    mu_p, logvar_p = prior(y_flat)

    elbo_terms = []
    for _ in range(L):
        eps = torch.randn_like(mu_q)
        z = mu_q + eps * torch.exp(0.5 * logvar_q)

        mu_x, logvar_x = dec_x(z)
        log_px = log_gaussian_likelihood(x_rep, mu_x, logvar_x)

        logits_tilde = dec_y(z)
        log_prob_ytilde = -F.cross_entropy(
            logits_tilde, y_tilde.unsqueeze(1).expand(B, topk).reshape(-1),
            reduction="none"
        )

        kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        elbo_k = log_px + log_prob_ytilde - kl
        elbo_terms.append(elbo_k)

    elbo_k_mc = torch.stack(elbo_terms, dim=0).mean(dim=0)  # (B*K,)
    elbo_k = elbo_k_mc.reshape(B, topk)
    elbo = (elbo_k * weight).sum(dim=-1).mean()

    loss = -elbo + lambda_sup * sup_loss
    return loss, {"elbo": elbo.detach(), "sup_ce": sup_loss.detach()}

# -----------------------
# 评估
# -----------------------
@torch.no_grad()
def evaluate_val(clf: nn.Module, val_feats: torch.Tensor, val_labels: torch.Tensor, batch: int = 1024) -> Tuple[float, float]:
    clf.eval()
    N = len(val_feats); correct = 0; total = 0; ce_sum = 0.0
    for i in range(0, N, batch):
        x = val_feats[i:i+batch].to(device)
        y = val_labels[i:i+batch].to(device)
        logits = clf(x)
        ce = F.cross_entropy(logits, y, reduction="sum")
        pred = logits.argmax(dim=-1)
        correct += (pred==y).sum().item()
        total += y.numel()
        ce_sum += ce.item()
    # Keep the CSV/metrics convention consistent with the other baselines:
    # accuracy is stored as a fraction in [0, 1].
    acc = correct / max(1,total)
    loss = ce_sum / max(1,total)
    return loss, acc

def plot_curves(csv_path: str, save_png_prefix: str):
    # Avoid requiring pandas for a simple two-column diagnostic plot.
    import csv
    epochs, val_acc, val_loss = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epochs.append(int(float(row["epoch"])))
            val_acc.append(float(row["val_acc"]))
            val_loss.append(float(row["val_loss"]))
    plt.figure()
    plt.plot(epochs, val_acc, marker="o")
    plt.xlabel("epoch"); plt.ylabel("val_acc (%)"); plt.title("Validation Accuracy")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{save_png_prefix}_val_acc.png", dpi=180); plt.close()

    plt.figure()
    plt.plot(epochs, val_loss, marker="o")
    plt.xlabel("epoch"); plt.ylabel("val_loss (CE)"); plt.title("Validation Loss (CE)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{save_png_prefix}_val_loss.png", dpi=180); plt.close()

# -----------------------
# ✅ 关键：构造 eval_acc.py 友好的线性头 state_dict
# -----------------------
def build_evalacc_friendly_state(clf: ClassifierHead) -> Dict[str, torch.Tensor]:
    """返回一个扁平 state_dict，至少包含 (C,D) 的 *.weight 与对应 bias。"""
    W = clf.fc.weight.detach().cpu()
    b = clf.fc.bias.detach().cpu()
    sd = {
        "linear.weight": W,   # 常用命名
        "linear.bias":   b,
        # 冗余一份常见别名，最大化兼容性
        "fc.weight":     W,
        "fc.bias":       b,
        "classifier.weight": W,
        "classifier.bias":   b,
    }
    return sd

# -----------------------
# 训练主程序
# -----------------------
def main():
    args = parse_args()
    save_experiment_config("CIFAR100_csgn", args)
    set_seed(args.seed)

    ensure_dir(args.val_dir)
    ensure_dir(os.path.dirname(args.best_ckpt) or ".")

    cache = load_cache_strict(args.cache_path)
    train_feats = cache["train_feats"]; val_feats = cache["val_feats"]; test_feats = cache["test_feats"]
    val_labels = cache["val_labels"];   test_labels = cache["test_labels"]
    in_dim = int(train_feats.size(-1))
    # Do not infer the vocabulary from test labels: some legacy caches use -1
    # for unavailable test labels.
    if isinstance(cache.get("clip_label_embeds"), torch.Tensor):
        num_classes = int(cache["clip_label_embeds"].shape[0])
    else:
        num_classes = int(max(cache["train_labels"].max().item(), val_labels.max().item()) + 1)

    y_obs, s = load_y_obs(args.obs_labels_path, n_train=len(train_feats))

    # batch_size = P*K
    batch_size = args.P * args.K
    train_ds = FeaturesWithNoisyLabels(train_feats, y_obs)
    sampler = BalancedPKSampler(labels=y_obs, P=args.P, K=args.K, drop_last=True)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True)

    # 模型
    clf = ClassifierHead(in_dim, num_classes).to(device)
    enc = EncoderZX(in_dim, num_classes, args.z_dim, hidden=256).to(device)
    prior = PriorZY(num_classes, args.z_dim).to(device)
    dec_x = DecoderX(args.z_dim, in_dim, logvar_x=-2.0).to(device)
    dec_y = DecoderYtilde(args.z_dim, num_classes).to(device)
    params = list(clf.parameters()) + list(enc.parameters()) + list(prior.parameters()) + list(dec_x.parameters()) + list(dec_y.parameters())

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # 日志
    csv_path = os.path.join(args.val_dir, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","val_loss","val_acc","train_elbo","train_sup_ce"])

    best_acc = -1.0; best_epoch = -1; wait = 0
    t0 = time.time()

    for epoch in range(1, args.epochs+1):
        clf.train(); enc.train(); prior.train(); dec_x.train(); dec_y.train()
        epoch_elbo = 0.0; epoch_sup = 0.0; n_batch = 0
        t_ep = time.time()
        for x, y_tilde in train_loader:
            x = x.to(device, non_blocking=True)
            y_tilde = y_tilde.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss, scalars = elbo_batch(
                x, y_tilde,
                model={"clf": clf, "enc": enc, "prior": prior, "dec_x": dec_x, "dec_y": dec_y},
                topk=args.topk_y, L=args.mc_L, lambda_sup=args.lambda_sup
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optim.step()
            epoch_elbo += scalars["elbo"].item()
            epoch_sup  += scalars["sup_ce"].item()
            n_batch += 1

        # 验证（线性头）
        val_loss, val_acc = evaluate_val(clf, val_feats.to(device), val_labels.to(device))
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f); w.writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.4f}",
                                           f"{epoch_elbo/max(1,n_batch):.4f}", f"{epoch_sup/max(1,n_batch):.4f}"])
        print(f"[Epoch {epoch:03d}] time={time.time()-t_ep:.1f}s | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc*100:.2f}% | elbo(avg)={epoch_elbo/max(1,n_batch):.4f} | "
              f"supCE(avg)={epoch_sup/max(1,n_batch):.4f}")

        # 早停（按 val_acc），保存 best_ckpt —— ✅ 兼容 eval_acc.py 的保存格式
        if val_acc > best_acc:
            best_acc = val_acc; best_epoch = epoch; wait = 0

            # 1) 构造扁平线性头 state_dict（含多种别名）
            evalacc_sd = build_evalacc_friendly_state(clf)

            # 2) 组装 payload：既保留完整模块参数，也放入 eval_acc 友好键位
            payload = {
                # eval_acc.py 优先使用的字段
                "state_dict": evalacc_sd,
                # 一些 Loader 直接把整个对象当 state_dict，用下面两个键也能匹配
                "linear.weight": evalacc_sd["linear.weight"],
                "linear.bias":   evalacc_sd["linear.bias"],
                # 可选：保留完整组件参数，便于复现/继续训练
                "clf_state": clf.state_dict(),
                "enc_state": enc.state_dict(),
                "prior_state": prior.state_dict(),
                "dec_x_state": dec_x.state_dict(),
                "dec_y_state": dec_y.state_dict(),
                # 元信息
                "in_dim": in_dim, "num_classes": num_classes,
                "epoch": epoch, "val_acc": float(best_acc), "val_loss": float(val_loss),
                "note": "Contains flat linear head for eval_acc.py (linear.weight/bias) + full module states."
            }

            torch.save(payload, args.best_ckpt)
            print(f"[CKPT] >>> Best so far (acc) saved to {args.best_ckpt}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EARLY-STOP] No improvement for {args.patience} epochs. Best epoch={best_epoch}, best acc={best_acc:.2f}%")
                break

    # 曲线（输出到 val_dir）
    plot_curves(csv_path, save_png_prefix=os.path.join(args.val_dir, "csgn_linear"))
    print(f"[DONE] total time = {time.time()-t0:.1f}s | best acc = {best_acc*100:.2f}% @ epoch {best_epoch}")

if __name__ == "__main__":
    main()
