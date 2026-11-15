# -*- coding: utf-8 -*-
"""
train_l2b_CIFAR100.py

运行示例：
python train_l2b_CIFAR100.py \
  --cache ./cache/CIFAR100/clip_ViTL14.pt \
  --obs_path ./cache/CIFAR100/cifar100_obs_labels.pt \
  --P 16 --K 16 --max_epochs 200 --warmup_epochs 3 \
  --meta_lr 1e-3 --patience 30
"""

import os, csv, time, argparse
from typing import Dict, Tuple, List, Iterator

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
# 通用
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _as_fp32_long(cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    fp32 = ["train_feats","val_feats","test_feats","clip_probs_train","clip_probs_val","clip_probs_test","clip_label_embeds"]
    i64  = ["train_labels","val_labels","test_labels","clip_argmax_train","clip_argmax_val","clip_argmax_test","train_idx_from_train50k","val_idx_from_train50k"]
    for k in fp32:
        if k in cache and isinstance(cache[k], torch.Tensor): cache[k] = cache[k].to(torch.float32)
    for k in i64:
        if k in cache and isinstance(cache[k], torch.Tensor): cache[k] = cache[k].long()
    return cache

def load_clip_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
    assert os.path.isfile(cache_path), f"Cache not found: {cache_path}"
    print(f"[CACHE] Loading from: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")
    cache = _as_fp32_long(cache)
    print("[CACHE] Loaded successfully.")
    return cache

def load_obs_labels(obs_path: str, expect_N: int) -> torch.Tensor:
    assert os.path.isfile(obs_path), f"obs_path not found: {obs_path}"
    obj = torch.load(obs_path, map_location="cpu")
    if isinstance(obj, dict) and "y_obs" in obj: y_obs = obj["y_obs"].long()
    elif torch.is_tensor(obj): y_obs = obj.long()
    else: raise RuntimeError("无法从 obs_path 解析 y_obs。需为包含 'y_obs' 的 dict 或 LongTensor。")
    assert y_obs.numel() == expect_N, f"y_obs 大小与训练集不符: {y_obs.numel()} != {expect_N}"
    return y_obs

# -----------------------
# 数据集 / 采样器（按 y_obs 做 P×K 均衡）
# -----------------------
class FeatureObsDataset(Dataset):
    def __init__(self, feats: torch.Tensor, y_obs: torch.Tensor):
        self.f = feats; self.y = y_obs.long()
    def __len__(self): return self.y.numel()
    def __getitem__(self, idx): return self.f[idx], int(self.y[idx])

class PKBatchSampler(Sampler[List[int]]):
    def __init__(self, labels_obs: torch.Tensor, P: int, K: int, seed: int = 42):
        super().__init__(None)
        self.labels = labels_obs.cpu().numpy()
        self.P, self.K, self.seed = int(P), int(K), int(seed)
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
            rng = random.Random(self.seed + c); rng.shuffle(idxs)
            self.ptrs[c] = {"idxs": idxs, "p": 0, "rng": rng}
    def __len__(self): return self.steps_per_epoch
    def _next_k(self, c, K):
        buf = []; s = self.ptrs[c]; idxs, p, rng = s["idxs"], s["p"], s["rng"]
        for _ in range(K):
            if p >= len(idxs): rng.shuffle(idxs); p = 0
            buf.append(idxs[p]); p += 1
        s["p"] = p; return buf
    def __iter__(self):
        import random
        rng = random.Random(self.seed)
        for _ in range(self.steps_per_epoch):
            chosen = rng.sample(self.classes, self.P) if len(self.classes)>=self.P else [rng.choice(self.classes) for _ in range(self.P)]
            batch_idx = []
            for c in chosen: batch_idx.extend(self._next_k(c, self.K))
            yield batch_idx

# -----------------------
# 模型 / 评估
# -----------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x): return self.fc(x)

@torch.no_grad()
def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 4096) -> Tuple[float, float]:
    head.eval()
    dl = DataLoader(TensorDataset(feats, labels), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ce = nn.CrossEntropyLoss(reduction="mean")
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in dl:
        xb = xb.to(device, dtype=torch.float32); yb = yb.to(device)
        logits = head(xb); loss = ce(logits, yb); pred = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((pred == yb).sum().item()); total += xb.size(0)
    return total_loss / max(1,total), total_correct / max(1,total)

class EarlyStopping:
    def __init__(self, metric: str = "acc", patience: int = 10, min_delta: float = 0.0):
        assert metric in ("loss", "acc")
        self.metric, self.patience, self.min_delta = metric, int(patience), float(min_delta)
        self.best = float("inf") if metric=="loss" else -float("inf"); self.num_bad = 0
    def step(self, val_loss: float, val_acc: float) -> bool:
        cur = val_loss if self.metric=="loss" else val_acc
        improved = (cur < self.best - self.min_delta) if self.metric=="loss" else (cur > self.best + self.min_delta)
        if improved: self.best = cur; self.num_bad = 0; return False
        self.num_bad += 1; print(f"早停已用耐心: {self.num_bad}"); return self.num_bad >= self.patience

def _save_val_curves(epochs, losses, accs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(); plt.plot(epochs, [a*100 for a in accs], marker="o"); plt.xlabel("Epoch"); plt.ylabel("Val Acc (%)")
    plt.title("Validation Accuracy"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"val_acc.png"), dpi=150); plt.close()
    plt.figure(); plt.plot(epochs, losses, marker="o"); plt.xlabel("Epoch"); plt.ylabel("Val CE Loss")
    plt.title("Validation Loss"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"val_loss.png"), dpi=150); plt.close()

# -----------------------
# Stabilized WeightNet: 输出 (w, mix)
# w: 样本总权重(>=0, 批内均值=1)；mix: per-sample 真/伪标签概率 (softmax)
# -----------------------
class WeightNet(nn.Module):
    def __init__(self, hidden: int = 64, tau: float = 1.0):
        super().__init__()
        self.mlp_shared = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(inplace=True),
        )
        self.head_mix = nn.Linear(hidden, 2)   # 产生 [a_i, b_i]
        self.head_w   = nn.Linear(hidden, 1)   # 产生 w_i (标量)
        self.softplus = nn.Softplus(beta=1.0)
        self.tau = tau
    def forward(self, loss_obs: torch.Tensor, loss_ps: torch.Tensor):
        x = torch.stack([loss_obs.detach(), loss_ps.detach()], dim=1)  # (B,2) 稳定高阶
        h = self.mlp_shared(x)                                        # (B,H)
        mix_logits = self.head_mix(h)                                  # (B,2)
        mix = F.softmax(mix_logits / self.tau, dim=1)                  # s1,s2∈(0,1), s1+s2=1
        w_raw = self.softplus(self.head_w(h).squeeze(1)) + 1e-6        # (B,) >=0
        # 批内均值归一（只对 w 做）：保持整体尺度≈1
        w = w_raw / (w_raw.mean().detach() + 1e-8)
        alpha = w * mix[:, 0]   # (B,)
        beta  = w * mix[:, 1]   # (B,)
        return alpha, beta, w, mix

# -----------------------
# 训练循环
# -----------------------
def make_cycle(it: Iterator, dl: DataLoader) -> Iterator:
    while True:
        try: batch = next(it)
        except StopIteration: it = iter(dl); batch = next(it)
        yield batch

def train(
    cache_path: str = "./cache/CIFAR100/clip_ViTL14.pt",
    obs_path: str = "./cache/CIFAR100/cifar100_obs_labels.pt",
    P: int = 16, K: int = 16,
    lr: float = 1e-3, lr_max: float = 5e-3, weight_decay: float = 1e-4,
    max_epochs: int = 200, warmup_epochs: int = 3,
    meta_lr: float = 1e-3, tau: float = 1.0,     # tau: 真/伪混合的温度
    ps_ramp_epochs: int = 3,                     # 伪标签斜坡时长（主阶段前几轮）
    print_every: int = 20,
    ckpt_path: str = "./outputs/CIFAR100/head_l2b.pt",
    best_ckpt_path: str = "./outputs/CIFAR100/head_l2b_best.pt",
    early_metric: str = "acc", patience: int = 10, min_delta: float = 0.0,
):
    set_seed(42)

    # 缓存与标签
    cache = load_clip_cache_strict(cache_path)
    train_feats = F.normalize(cache["train_feats"].to(torch.float32), dim=-1)
    val_feats   = F.normalize(cache["val_feats"].to(torch.float32),   dim=-1)
    train_labels_true = cache["train_labels"].long()
    val_labels_true   = cache["val_labels"].long()

    N, D = train_feats.shape
    C = int(max(train_labels_true.max().item(), val_labels_true.max().item()) + 1)
    y_obs = load_obs_labels(obs_path, expect_N=N)

    print(f"[INFO] Ntrain={N}, Nval={val_labels_true.numel()}, D={D}, C={C} | P={P}, K={K}, batch={P*K}")
    print(f"[INFO] obs_path: {obs_path} | y_obs ∈ [{int(y_obs.min())}, {int(y_obs.max())}]")

    # DataLoader
    train_ds = FeatureObsDataset(train_feats, y_obs)
    batch_sampler = PKBatchSampler(labels_obs=y_obs, P=P, K=K, seed=42)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    val_ds = TensorDataset(val_feats, val_labels_true)
    val_loader = DataLoader(val_ds, batch_size=P*K, shuffle=True, num_workers=2, pin_memory=True)
    val_iter = make_cycle(iter(val_loader), val_loader)

    # 线性头 & 优化器
    head = LinearHead(D, C).to(device).float()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr_max, total_steps=max(1, steps_per_epoch * max_epochs),
        pct_start=0.15, anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0
    )

    # 权重网 & 优化器
    wnet = WeightNet(hidden=64, tau=tau).to(device)
    opt_w = torch.optim.Adam(wnet.parameters(), lr=meta_lr)

    ce_none = nn.CrossEntropyLoss(reduction="none")
    stopper = EarlyStopping(metric=early_metric, patience=patience, min_delta=min_delta)

    # 记录
    val_dir = "./val/CIFAR100"; os.makedirs(val_dir, exist_ok=True)
    csv_path = os.path.join(val_dir, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f: csv.writer(f).writerow(["epoch","val_loss","val_acc"])
    l2b_csv = os.path.join(val_dir, "l2b_stats.csv")
    with open(l2b_csv, "w", newline="") as f: csv.writer(f).writerow(["epoch","mix_ps_mean","w_mean","w_cv"])

    val_epochs, val_losses, val_accs = [], [], []
    print("[TRAIN] Start L2B (stabilized) ...")

    for epoch in range(1, max_epochs + 1):
        head.train(); t0 = time.time()
        running_loss, seen = 0.0, 0
        mix_ps_sum, w_sum, wcv_sum, stat_cnt = 0.0, 0.0, 0.0, 0

        # 伪标签斜坡系数（进入主阶段后从0→1）
        if epoch <= warmup_epochs: gamma_ps = 0.0
        else:
            k = epoch - warmup_epochs
            gamma_ps = min(1.0, k / max(1, ps_ramp_epochs))

        for ib, (f, yb) in enumerate(train_loader, start=1):
            f = f.to(device, dtype=torch.float32); yb = yb.to(device)

            # ---------- 前向 ----------
            logits = head(f)
            with torch.no_grad(): yhat = logits.argmax(dim=-1)
            loss_obs = ce_none(logits, yb)
            loss_ps  = ce_none(logits, yhat)

            # ---------- Warm-up：只用观测标签 ----------
            if epoch <= warmup_epochs:
                loss = loss_obs.mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                opt.step(); scheduler.step()
                running_loss += float(loss.item()); seen += f.size(0)
                if ib % print_every == 0:
                    print(f"[Epoch {epoch:03d}] batch {ib:04d}/{len(train_loader):04d} | stage=warmup | seen={seen:6d} | loss(avg)={running_loss/ib:.5f}")
                continue

            # ---------- Step-1: 元更新（fast weights 可微） ----------
            alpha, beta, w, mix = wnet(loss_obs, loss_ps)     # α+β = w
            # 伪标签斜坡
            beta = beta * gamma_ps

            L_train_meta = (alpha * loss_obs + beta * loss_ps).mean()
            gw, gb = torch.autograd.grad(L_train_meta, [head.fc.weight, head.fc.bias], create_graph=True, retain_graph=True)
            lr_now = scheduler.get_last_lr()[0]
            fast_w = head.fc.weight - lr_now * gw
            fast_b = head.fc.bias  - lr_now * gb

            # 验证损失（用 fast weights）
            xv, yv = next(val_iter)
            xv = xv.to(device, dtype=torch.float32); yv = yv.to(device)
            logits_v = F.linear(xv, fast_w, fast_b)
            L_val = F.cross_entropy(logits_v, yv, reduction="mean")

            # 仅更新权重网
            opt_w.zero_grad(set_to_none=True)
            grads_w = torch.autograd.grad(L_val, wnet.parameters(), create_graph=False, retain_graph=False)
            for p, g in zip(wnet.parameters(), grads_w): p.grad = g
            torch.nn.utils.clip_grad_norm_(wnet.parameters(), 5.0)
            opt_w.step()

            # ---------- Step-2: 真实更新（重前向） ----------
            logits2 = head(f)
            with torch.no_grad(): yhat2 = logits2.argmax(dim=-1)
            loss_obs2 = ce_none(logits2, yb)
            loss_ps2  = ce_none(logits2, yhat2)

            alpha2, beta2, w2, mix2 = wnet(loss_obs2, loss_ps2)
            beta2 = beta2 * gamma_ps
            L_train_real = (alpha2 * loss_obs2 + beta2 * loss_ps2).mean()

            opt.zero_grad(set_to_none=True); L_train_real.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step(); scheduler.step()

            running_loss += float(L_train_real.item()); seen += f.size(0)

            # 统计：伪标签比例/样本权 w 的均值与离散度
            with torch.no_grad():
                mix_ps = (beta2 / (alpha2 + beta2 + 1e-8)).mean().item()   # = mix[:,1] 的均值（*γ后）
                w_mean = w2.mean().item()
                w_cv   = (w2.std() / (w2.mean() + 1e-8)).item()
                mix_ps_sum += mix_ps; w_sum += w_mean; wcv_sum += w_cv; stat_cnt += 1

            if ib % print_every == 0:
                print(f"[Epoch {epoch:03d}] batch {ib:04d}/{len(train_loader):04d} | stage=main | seen={seen:6d} | "
                      f"mix_ps={mix_ps:.3f} γ={gamma_ps:.2f} | w(mean)={w_mean:.2f} w(cv)={w_cv:.2f} | loss(avg)={running_loss/ib:.5f}")

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] time={dt:.1f}s | avg_loss={running_loss/max(1,len(train_loader)):.5f}")

        # ---------- 验证 ----------
        val_loss, val_acc = evaluate(head, val_feats, val_labels_true, batch_size=4096)
        print(f"[VAL] epoch={epoch:03d} | val_loss={val_loss:.5f} | val_acc={val_acc*100:.2f}%")

        val_epochs.append(epoch); val_losses.append(val_loss); val_accs.append(val_acc)
        with open(csv_path, "a", newline="") as f: csv.writer(f).writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.6f}"])
        _save_val_curves(val_epochs, val_losses, val_accs, val_dir)
        print(f"[VAL-SAVE] CSV -> {csv_path} | PNG -> {val_dir}/val_acc.png, val_loss.png")

        if stat_cnt > 0:
            with open(l2b_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{(mix_ps_sum/stat_cnt):.6f}", f"{(w_sum/stat_cnt):.6f}", f"{(wcv_sum/stat_cnt):.6f}"])

        # 保存 & 早停
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(head.state_dict(), ckpt_path)
        is_best = (early_metric=="loss" and val_loss <= getattr(stopper,"best",float("inf"))) or \
                  (early_metric=="acc"  and val_acc  >= getattr(stopper,"best",-float("inf")))
        if is_best and best_ckpt_path:
            torch.save(head.state_dict(), best_ckpt_path)
            print(f"[CKPT] >>> Best so far ({early_metric}) saved to {best_ckpt_path}")
        if stopper.step(val_loss, val_acc):
            print(f"[EARLY STOP] No improvement in {early_metric} for {stopper.patience} epochs. Stop."); break

    print("[DONE] L2B (stabilized) finished.")

# -----------------------
# main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default="./cache/CIFAR100/clip_ViTL14.pt")
    p.add_argument("--obs_path", type=str, default="./cache/CIFAR100/cifar100_obs_labels.pt")
    p.add_argument("--P", type=int, default=16); p.add_argument("--K", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--lr_max", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=200); p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--meta_lr", type=float, default=1e-3); p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--ps_ramp_epochs", type=int, default=3)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--ckpt", type=str, default="./outputs/CIFAR100/temp.pt")
    p.add_argument("--best_ckpt", type=str, default="./outputs/CIFAR100/head_l2b_best.pt")
    p.add_argument("--early_metric", type=str, default="acc", choices=["loss","acc"])
    p.add_argument("--patience", type=int, default=8); p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args(); set_seed(args.seed)
    train(
        cache_path=args.cache, obs_path=args.obs_path,
        P=args.P, K=args.K,
        lr=args.lr, lr_max=args.lr_max, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, warmup_epochs=args.warmup_epochs,
        meta_lr=args.meta_lr, tau=args.tau, ps_ramp_epochs=args.ps_ramp_epochs,
        print_every=args.print_every,
        ckpt_path=args.ckpt, best_ckpt_path=args.best_ckpt,
        early_metric=args.early_metric, patience=args.patience, min_delta=args.min_delta,
    )
