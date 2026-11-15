# -*- coding: utf-8 -*-
"""
train_csgn_CIFAR100.py

使用示例：
python train_csgn_CIFAR100.py \
  --cache_path ./cache/CIFAR100/clip_ViTL14.pt \
  --obs_labels_path ./cache/CIFAR100/cifar100_obs_labels.pt \
  --epochs 100 --patience 8 \
  --batch_size 512 --P 32 --K 16 \
  --z_dim 64 --lambda_sup 0.1 \
  --logdir ./val/CIFAR100 --outdir ./outputs/CIFAR100 --tag head_csgn
"""

import os, time, math, argparse, csv
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

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
# 严格读取缓存（与 extract_CLIP_cifar100.py 一致）
# -----------------------
def load_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")

    required = ["train_feats","train_labels","val_feats","val_labels","test_feats","test_labels"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"缓存缺少必需键: {missing}。请确认来自 extract_CLIP_cifar100.py。")

    # 统一 dtype
    for k in ["train_feats","val_feats","test_feats","clip_label_embeds"]:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(torch.float32)
    for k in ["train_labels","val_labels","test_labels"]:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(torch.long)

    # 友好提示（不是必须）
    for k in ["clip_probs_train","clip_probs_val","clip_probs_test",
              "clip_argmax_train","clip_argmax_val","clip_argmax_test",
              "train_idx_from_train50k","val_idx_from_train50k",
              "class_names","image_size","model_name","seed","n_val_per_class"]:
        if k in data:
            pass  # 仅为保持与缓存一致；不做强依赖

    return data

# -----------------------
# 读取 y_obs（与 make_obs_labels.py 一致）
# -----------------------
def load_y_obs(obs_labels_path: str, n_train: int):
    """
    期望结构：
    {
      "y_obs": LongTensor (Ntr,),
      "s":     LongTensor (Ntr,)  # 可选，不区分可信/不可信时忽略
      "meta":  dict                # 可选
    }
    也兼容直接 LongTensor 或包含 y_obs 的任意 dict。
    """
    if not os.path.exists(obs_labels_path):
        raise FileNotFoundError(f"Obs labels not found: {obs_labels_path}")
    obj = torch.load(obs_labels_path, map_location="cpu")

    y_obs = None; s = None; meta = None
    if isinstance(obj, torch.Tensor):
        y_obs = obj
    elif isinstance(obj, dict):
        if "y_obs" in obj: y_obs = obj["y_obs"]
        elif "obs_labels" in obj: y_obs = obj["obs_labels"]
        else:
            # 兜底：找第一个 long 向量
            for v in obj.values():
                if isinstance(v, torch.Tensor) and v.dtype in (torch.int64, torch.int32) and v.dim()==1:
                    y_obs = v; break
        if "s" in obj and isinstance(obj["s"], torch.Tensor):
            s = obj["s"]
        if "meta" in obj and isinstance(obj["meta"], dict):
            meta = obj["meta"]
    else:
        raise RuntimeError(f"{obs_labels_path} 格式不支持。")

    if y_obs is None:
        raise RuntimeError(f"{obs_labels_path} 未找到 y_obs。期望键 'y_obs'。")
    y_obs = y_obs.to(torch.long).view(-1)
    if y_obs.numel() != n_train:
        raise RuntimeError(f"y_obs 长度 {y_obs.numel()} 与训练集样本数 {n_train} 不符。")

    # s 可选
    if s is not None:
        s = s.to(torch.long).view(-1)
        if s.numel() != n_train:
            print(f"[WARN] s 长度 {s.numel()} != 训练集 {n_train}，忽略 s。")
            s = None

    # 打印 meta（若存在）
    if meta is not None:
        top_pct = meta.get("top_outlier_pct", None)
        from_cache = meta.get("from_cache", None)
        print(f"[OBS] meta: top_outlier_pct={top_pct} | from_cache={from_cache}")

    # 简报
    if s is not None:
        num_trusted = int((s==0).sum().item())
        print(f"[OBS] y_obs loaded. trusted s=0: {num_trusted}  |  untrusted s=1: {int(s.numel())-num_trusted}")
    else:
        print(f"[OBS] y_obs loaded. (no s provided)")
    return y_obs, s

# -----------------------
# P×K 均衡采样（按 y_obs）
# -----------------------
class BalancedPKSampler(Sampler):
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
        return self.batches_per_epoch * self.batch_size if self.drop_last else self.num_samples

    def __iter__(self):
        rng = np.random.default_rng()
        batches = []
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
            batches.extend(batch)
        return iter(batches)

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
    acc = 100.0 * correct / max(1,total)
    loss = ce_sum / max(1,total)
    return loss, acc

def plot_curves(csv_path: str, save_png_prefix: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df["epoch"], df["val_acc"], marker="o")
    plt.xlabel("epoch"); plt.ylabel("val_acc (%)"); plt.title("Validation Accuracy")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{save_png_prefix}_val_acc.png", dpi=180); plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["val_loss"], marker="o")
    plt.xlabel("epoch"); plt.ylabel("val_loss (CE)"); plt.title("Validation Loss (CE)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{save_png_prefix}_val_loss.png", dpi=180); plt.close()

# -----------------------
# 主程序
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--obs_labels_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--P", type=int, default=32)
    parser.add_argument("--K", type=int, default=16)

    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--lambda_sup", type=float, default=0.1)
    parser.add_argument("--topk_y", type=int, default=5)
    parser.add_argument("--mc_L", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--logdir", type=str, default="./val/CIFAR100")
    parser.add_argument("--outdir", type=str, default="./outputs/CIFAR100")
    parser.add_argument("--tag", type=str, default="head_csgn_linear")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.logdir); ensure_dir(args.outdir)

    # 读缓存（严格与 extract_CLIP_cifar100.py 对齐）
    cache = load_cache_strict(args.cache_path)
    train_feats = cache["train_feats"]; val_feats = cache["val_feats"]; test_feats = cache["test_feats"]
    val_labels = cache["val_labels"];   test_labels = cache["test_labels"]
    in_dim = int(train_feats.size(-1))
    num_classes = int(max(val_labels.max().item(), test_labels.max().item()) + 1)

    # 读 y_obs（与 make_obs_labels.py 对齐）
    y_obs, s = load_y_obs(args.obs_labels_path, n_train=len(train_feats))

    # 数据/采样
    if args.batch_size != args.P * args.K:
        print(f"[WARN] batch_size {args.batch_size} != P*K {args.P*args.K}，自动改为 P*K。")
        args.batch_size = args.P * args.K
    train_ds = FeaturesWithNoisyLabels(train_feats, y_obs)
    sampler = BalancedPKSampler(labels=y_obs, P=args.P, K=args.K, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    # 模型
    clf = ClassifierHead(in_dim, num_classes).to(device)
    enc = EncoderZX(in_dim, num_classes, args.z_dim, hidden=256).to(device)
    prior = PriorZY(num_classes, args.z_dim).to(device)
    dec_x = DecoderX(args.z_dim, in_dim, logvar_x=-2.0).to(device)
    dec_y = DecoderYtilde(args.z_dim, num_classes).to(device)
    params = list(clf.parameters()) + list(enc.parameters()) + list(prior.parameters()) + list(dec_x.parameters()) + list(dec_y.parameters())

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # 训练
    best_acc = -1.0; best_epoch = -1; wait = 0
    csv_path = os.path.join(args.logdir, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","val_loss","val_acc","train_elbo","train_sup_ce"])

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

        # 验证：与主流水线一致，使用干净 val
        val_loss, val_acc = evaluate_val(clf, val_feats.to(device), val_labels.to(device))
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f); w.writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.4f}", f"{epoch_elbo/max(1,n_batch):.4f}", f"{epoch_sup/max(1,n_batch):.4f}"])
        print(f"[Epoch {epoch:03d}] time={time.time()-t_ep:.1f}s | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}% | elbo(avg)={epoch_elbo/max(1,n_batch):.4f} | supCE(avg)={epoch_sup/max(1,n_batch):.4f}")

        # 早停（按 val_acc）
        if val_acc > best_acc:
            best_acc = val_acc; best_epoch = epoch; wait = 0
            out_path = os.path.join(args.outdir, f"{args.tag}_best.pt")
            torch.save({
                "clf_state": clf.state_dict(),
                "enc_state": enc.state_dict(),
                "prior_state": prior.state_dict(),
                "dec_x_state": dec_x.state_dict(),
                "dec_y_state": dec_y.state_dict(),
                "in_dim": in_dim, "z_dim": args.z_dim, "num_classes": num_classes,
                "epoch": epoch, "val_acc": best_acc, "val_loss": val_loss
            }, out_path)
            print(f"[CKPT] >>> Best so far (acc) saved to {out_path}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EARLY-STOP] No improvement for {args.patience} epochs. Best epoch={best_epoch}, best acc={best_acc:.2f}%")
                break

    # 曲线
    plot_curves(csv_path, save_png_prefix=os.path.join(args.logdir, "csgn_linear"))
    print(f"[DONE] total time = {time.time()-t0:.1f}s | best acc = {best_acc:.2f}% @ epoch {best_epoch}")

if __name__ == "__main__":
    main()
