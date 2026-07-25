# -*- coding: utf-8 -*-
"""
train_auc_ce.py

- 从缓存读取特征与标签
- 仅训练线性头；P×K 均衡采样；验证集(CE)早停；保存 val 曲线与最佳 ckpt
- 目标函数：Loss =  L_CE(trusted) + lam * L_AUC(untrusted)
  * L_CE(trusted)：交叉熵仅对 s=0 可信样本（标签用 y_true）
  * L_AUC(untrusted)：OVR pairwise logistic AUC 仅对 s=1 不可信样本（alpha=0.5, tau=0.5）
"""

import os, time, argparse, csv
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

# ----------------------- 参数 -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default=str(clip_cache("CIFAR100")))
    p.add_argument("--obs_labels_path", type=str, default=str(observed_labels_cache("CIFAR100")))
    p.add_argument("--P", type=int, default=32)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_max", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=200)

    
    p.add_argument("--lam", type=float, default=1)
    p.add_argument("--best_ckpt", type=str, default=str(experiment_checkpoint("CIFAR100_auc_ce")))
    p.add_argument("--val_dir", type=str, default=str(experiment_logs("CIFAR100_auc_ce")))
    p.add_argument("--experiment_name", type=str, default=None,
                   help="Name used for the saved config snapshot.")


    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ----------------------- 通用/缓存读取 -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_clip_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    print(f"[CACHE] Loading from: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")
    fp32 = ["train_feats","val_feats","test_feats",
            "clip_probs_train","clip_probs_val","clip_probs_test",
            "clip_label_embeds"]
    i64  = ["train_labels","val_labels","test_labels",
            "clip_argmax_train","clip_argmax_val","clip_argmax_test",
            "train_idx_from_train50k","val_idx_from_train50k"]
    for k in fp32:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(torch.float32)
    for k in i64:
        if k in data and isinstance(data[k], torch.Tensor):
            data[k] = data[k].long()
    print("[CACHE] Loaded successfully.")
    return data

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

# ----------------------- 数据集/采样器/线性头 -----------------------
class FeatureDataset(Dataset):
    def __init__(self, feats, y_true, y_obs, s, p_clip):
        self.f, self.y_true, self.y_obs, self.s, self.p_clip = feats, y_true, y_obs, s, p_clip
    def __len__(self): return self.f.size(0)
    def __getitem__(self, i):
        return self.f[i], int(self.y_true[i]), int(self.y_obs[i]), int(self.s[i]), self.p_clip[i]

class PKBatchSampler(Sampler[List[int]]):
    """每个 batch 采样 P 个类别、每类 K 个样本 -> batch_size = P*K；按 y_obs 做类均衡。"""
    def __init__(self, labels_obs: torch.Tensor, P: int, K: int, seed: int = 42):
        super().__init__(None)
        import random, numpy as np
        self.labels = labels_obs.cpu().numpy(); self.P=int(P); self.K=int(K); self.seed=int(seed)
        self.class_to_indices = {}
        for c in np.unique(self.labels):
            idx = np.where(self.labels == c)[0].tolist()
            self.class_to_indices[int(c)] = idx
        self.classes = list(self.class_to_indices.keys())
        self.N = len(self.labels); self.batch_size = self.P*self.K
        self.steps_per_epoch = max(1, self.N//self.batch_size)
        self.ptrs = {}
        for c, idxs in self.class_to_indices.items():
            rng = random.Random(self.seed + c); rng.shuffle(idxs)
            self.ptrs[c] = {"idxs": idxs, "p": 0, "rng": rng}
    def __len__(self): return self.steps_per_epoch
    def _next_k(self, c, K):
        buf=[]; state=self.ptrs[c]; idxs,p,rng=state["idxs"],state["p"],state["rng"]
        for _ in range(K):
            if p >= len(idxs): rng.shuffle(idxs); p=0
            buf.append(idxs[p]); p+=1
        state["p"]=p; return buf
    def __iter__(self):
        import random
        rng = random.Random(self.seed)
        for _ in range(self.steps_per_epoch):
            chosen = rng.sample(self.classes, self.P) if len(self.classes)>=self.P else [rng.choice(self.classes) for _ in range(self.P)]
            batch_idx=[]
            for c in chosen: batch_idx.extend(self._next_k(c, self.K))
            yield batch_idx

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x): return self.fc(x)

# ----------------------- AUC（仅用于 s=1） -----------------------
def auc_noise_only_a11_loss(logits, clip_probs, s, alpha=0.5, tau=0.5, eps: float = 1e-12):
    """
    OVR pairwise logistic AUC；仅对 s==1 的样本成对。
    q = 0.5 * p_model + 0.5 * p_clip（alpha=0.5 固定）；tau=0.5 固定。
    """
    device = logits.device; C = logits.size(1)
    idx = torch.nonzero(s == 1, as_tuple=False).squeeze(1); m = idx.numel()
    if m < 2:
        return logits.new_tensor(0.0, requires_grad=True)
    lo = logits[idx]                          # (m, C)
    p_clip = clip_probs[idx].detach()         # (m, C)
    p_model = F.softmax(lo, dim=-1).detach()           # (m, C)
    q = 0.5 * p_model + 0.5 * p_clip          # alpha=0.5
    offdiag = ~torch.eye(m, dtype=torch.bool, device=device)
    total_loss = lo.new_tensor(0.0); total_w = lo.new_tensor(0.0)
    for i in range(C):
        s_i = lo[:, i]; q_i = q[:, i]
        diff = s_i[:, None] - s_i[None, :]
        w_posneg = (q_i[:, None]) * (1.0 - q_i)[None, :]
        w_negpos = (1.0 - q_i[:, None]) * (q_i)[None, :]
        ell_posneg = F.softplus(-diff / tau)  # tau=0.5
        ell_negpos = F.softplus( diff / tau)
        w_sum = (w_posneg + w_negpos)
        loss_mat = w_posneg * ell_posneg + w_negpos * ell_negpos
        total_loss = total_loss + (loss_mat * offdiag).sum()
        total_w    = total_w    + (w_sum    * offdiag).sum()
    return total_loss / (total_w + eps)

# ----------------------- 评估/早停/画图 -----------------------
@torch.no_grad()
def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048) -> Tuple[float, float]:
    head.eval(); ds = TensorDataset(feats, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ce = nn.CrossEntropyLoss(reduction="sum"); total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in dl:
        xb = xb.to(device, dtype=torch.float32); yb = yb.to(device)
        logits = head(xb); loss = ce(logits, yb); pred = logits.argmax(dim=-1)
        total_loss += float(loss.item()); total_correct += int((pred == yb).sum().item()); total += xb.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)

class EarlyStopping:
    """按 val_acc 早停（和原脚本一致：step(val_loss, val_acc) 接口）"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience, self.min_delta = int(patience), float(min_delta)
        self.best = -float("inf"); self.num_bad = 0
    def step(self, val_loss: float, val_acc: float) -> bool:
        improved = (val_acc > self.best + self.min_delta)
        if improved: self.best = val_acc; self.num_bad = 0; return False
        self.num_bad += 1; print(f"早停已用耐心:{self.num_bad}"); return self.num_bad >= self.patience

def _save_val_curves(epochs, losses, accs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # val_acc（%）
    plt.figure(); plt.plot(epochs, [a * 100.0 for a in accs], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Val Acc (%)"); plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_acc.png"), dpi=150); plt.close()
    # val_loss
    plt.figure(); plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Val CE Loss"); plt.title("Validation Loss")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_loss.png"), dpi=150); plt.close()

# ----------------------- 训练（Loss = AUC + lam * CE） -----------------------
def train(
    cache_path: str,
    obs_labels_path: str,
    P: int, K: int,
    lr: float, lr_max: float, weight_decay: float,
    max_epochs: int,
    lam: float,
    print_every: int,
    best_ckpt_path: str,
    patience: int, min_delta: float,
    val_dir: str,               
):
    set_seed(42); lam = float(lam)

    # 0) 缓存与标准化
    cache = load_clip_cache_strict(cache_path)
    train_feats = F.normalize(cache["train_feats"].to(torch.float32), dim=-1)
    val_feats   = F.normalize(cache["val_feats"].to(torch.float32),   dim=-1)
    train_labels, val_labels = cache["train_labels"].long(), cache["val_labels"].long()
    clip_probs_train = cache["clip_probs_train"].to(torch.float32)

    N, D = train_feats.shape
    C = int(max(train_labels.max().item(), val_labels.max().item())) + 1
    print(f"[INFO] Ntrain={N}, Nval={val_labels.numel()}, D={D}, C={C} | P={P}, K={K}, batch={P*K}")
    print(f"[OBJECTIVE] Loss = CE(trusted) + lam * AUC(untrusted) | lam={lam:.2f} | tau=0.5 | early_metric=acc")

    # 1) 读取 obs_path（y_obs, s）
    obs = load_obs_labels(obs_labels_path)
    y_obs_after, s_flags = obs["y_obs"], obs["s"]
    if y_obs_after.numel() != N or s_flags.numel() != N:
        raise RuntimeError(f"Obs length mismatch: feats N={N}, y_obs={y_obs_after.numel()}, s={s_flags.numel()}")

    # 2) DataLoader（用 y_obs 做均衡）
    train_ds = FeatureDataset(train_feats, train_labels, y_obs_after, s_flags, clip_probs_train)
    batch_sampler = PKBatchSampler(labels_obs=y_obs_after, P=P, K=K, seed=42)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    # 3) 线性头 + 优化器/调度器
    head = LinearHead(D, C).to(device).float()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr_max, total_steps=max(1, steps_per_epoch*max_epochs),
        pct_start=0.15, anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0
    )
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    ce_criterion = nn.CrossEntropyLoss(reduction="mean")

    # 记录
    os.makedirs(val_dir, exist_ok=True)
    csv_path = os.path.join(val_dir, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","val_loss","val_acc"])
    val_epochs, val_losses, val_accs = [], [], []

    print("[TRAIN] Start ...")
    for epoch in range(1, max_epochs + 1):
        head.train(); t0 = time.time()
        running = 0.0; seen = 0

        for ib, batch in enumerate(train_loader, start=1):
            f, y_true, y_obs, s, p_clip = batch
            f = f.to(device, dtype=torch.float32)
            y_true = y_true.to(device); s = s.to(device)
            p_clip = p_clip.to(device, dtype=torch.float32)

            logits = head(f); B = f.size(0)
            mask_t = (s == 0)  # 可信

            # CE 仅对可信样本
            if mask_t.any():
                ce_trusted = ce_criterion(logits[mask_t], y_true[mask_t])
            else:
                ce_trusted = logits.sum()*0.0

            # AUC 仅对不可信样本
            auc_untrusted = auc_noise_only_a11_loss(
                logits=logits, clip_probs=p_clip, s=s, alpha=0.5, tau=0.5
            )

            # Loss = CE + lam * AUC 
            loss = ce_trusted + lam * auc_untrusted

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            opt.step(); sched.step()

            running += float(loss.item()); seen += B
            if ib % print_every == 0:
                print(
                    f"[Epoch {epoch:03d}] {ib:04d}/{len(train_loader):04d} | "
                    f"B={B} Bt={int(mask_t.sum())} | lam={lam:.2f} | "
                    f"CE={float(ce_trusted.item()):.5f} | AUC={float(auc_untrusted.item()):.5f} | "
                    f"Loss(avg)={running/ib:.5f}"
                )

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] time={dt:.1f}s | avg_loss={running/max(1,len(train_loader)):.5f}")

        # 验证与记录
        val_loss, val_acc = evaluate(head, val_feats, val_labels, batch_size=2048)
        print(f"[VAL] epoch={epoch:03d} | val_loss={val_loss:.5f} | val_acc={val_acc*100:.2f}%")
        val_epochs.append(epoch); val_losses.append(val_loss); val_accs.append(val_acc)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.6f}"])
        _save_val_curves(val_epochs, val_losses, val_accs, val_dir)
        print(f"[VAL-SAVE] CSV -> {csv_path} | PNG -> {val_dir}/val_acc.png, val_loss.png")

        # 仅保存最佳 ckpt（early_metric 固定为 acc）
        is_best = val_acc >= getattr(stopper, "best", -float('inf'))
        if is_best and best_ckpt_path:
            os.makedirs(os.path.dirname(best_ckpt_path) or ".", exist_ok=True)
            torch.save(head.state_dict(), best_ckpt_path)
            print(f"[CKPT] >>> Best so far (acc) saved to {best_ckpt_path}")
        if best_ckpt_path:
            last_ckpt_path = os.path.join(os.path.dirname(best_ckpt_path) or ".", "last.ckpt")
            torch.save(head.state_dict(), last_ckpt_path)
            print(f"[CKPT] Last checkpoint saved to {last_ckpt_path}")

        if stopper.step(val_loss, val_acc):
            print(f"[EARLY STOP] No improvement in acc for {stopper.patience} epochs.")
            break
    print("[DONE] Training finished.")

# ----------------------- main -----------------------
if __name__ == "__main__":
    args = parse_args()
    config_name = args.experiment_name or Path(args.best_ckpt).parent.name or "CIFAR100_auc_ce"
    save_experiment_config(config_name, args)
    set_seed(args.seed)
    train(
        cache_path=args.cache, obs_labels_path=args.obs_labels_path,
        P=args.P, K=args.K,
        lr=args.lr, lr_max=args.lr_max, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        lam=args.lam,
        print_every=args.print_every,
        best_ckpt_path=args.best_ckpt,
        patience=args.patience, min_delta=args.min_delta,
        val_dir=args.val_dir,  
    )
