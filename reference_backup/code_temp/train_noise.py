# -*- coding: utf-8 -*-
"""
train_noise.py

仅使用“不可信样本（s=1）”训练：
- 主目标：A11 OVR pairwise logistic AUC（q = alpha*p_model + (1-alpha)*p_clip）
- 稳定器：
  (a) KD：CE(p_clip || p_model)，权重从 kd_start -> kd_final（逐步衰减）
  (b) 置信门控：仅对 max(p_clip) >= conf_t 的样本计入 AUC/KD，conf_t 从 conf_start -> conf_final 线性下降
- 其余：复用特征读取、P×K 采样、验证/早停、val 曲线与 CSV 写入
"""

import os, math, time, argparse, csv
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------- utils -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_clip_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(cache_path): raise FileNotFoundError(cache_path)
    print(f"[CACHE] Loading from: {cache_path}")
    data = torch.load(cache_path, map_location="cpu")
    fp32 = ["train_feats","val_feats","test_feats","clip_probs_train","clip_probs_val","clip_probs_test","clip_label_embeds"]
    longs = ["train_labels","val_labels","test_labels","clip_argmax_train","clip_argmax_val","clip_argmax_test","train_idx_from_train50k","val_idx_from_train50k"]
    for k in fp32:
        if k in data and isinstance(data[k], torch.Tensor): data[k] = data[k].to(torch.float32)
    for k in longs:
        if k in data and isinstance(data[k], torch.Tensor): data[k] = data[k].long()
    print("[CACHE] Loaded successfully.")
    return data

def outlier_correction_by_class_centers(feats, y_true, y_pseudo, top_pct=0.30):
    N, D = feats.shape
    C = int(y_true.max().item()) + 1
    feats = F.normalize(feats, dim=-1)
    y_obs = y_pseudo.clone()
    s = torch.ones(N, dtype=torch.long)
    print("[STEP 1] Computing class centers and outliers ...")
    for c in range(C):
        idx = torch.nonzero(y_pseudo == c, as_tuple=False).flatten()
        if idx.numel() == 0: continue
        F_c = feats[idx]
        center = F.normalize(F_c.mean(dim=0, keepdim=True), dim=-1)
        cos_sim = (F_c @ center.t()).squeeze(1)
        dist = 1.0 - cos_sim
        k = max(1, int(math.ceil(len(idx)*top_pct)))
        _, topk_ind = torch.topk(dist, k, largest=True, sorted=False)
        out_idx = idx[topk_ind]
        y_obs[out_idx] = y_true[out_idx]; s[out_idx] = 0
        print(f"[STEP 1] class={c:03d} | size={len(idx):4d} | corrected={k:4d}")
    print(f"[STEP 1] Trusted(s=0)={(s==0).sum().item()} | Untrusted(s=1)={(s==1).sum().item()}")
    return y_obs, s

class FeatureDataset(Dataset):
    def __init__(self, feats, y_true, y_obs, s, p_clip):
        self.f, self.y_true, self.y_obs, self.s, self.p_clip = feats, y_true, y_obs, s, p_clip
    def __len__(self): return self.f.size(0)
    def __getitem__(self, i):
        return self.f[i], int(self.y_true[i]), int(self.y_obs[i]), int(self.s[i]), self.p_clip[i]

class PKBatchSampler(Sampler[List[int]]):
    def __init__(self, labels_obs: torch.Tensor, P: int, K: int, seed: int = 42):
        super().__init__(None)
        self.labels = labels_obs.cpu().numpy(); self.P=int(P); self.K=int(K); self.seed=int(seed)
        self.class_to_indices = {}
        for c in np.unique(self.labels):
            idx = np.where(self.labels==c)[0].tolist(); self.class_to_indices[int(c)] = idx
        self.classes = list(self.class_to_indices.keys())
        self.N = len(self.labels); self.batch_size=self.P*self.K
        self.steps_per_epoch = max(1, self.N//self.batch_size)
        import random
        self.ptrs={}
        for c, idxs in self.class_to_indices.items():
            rng = random.Random(self.seed+c); rng.shuffle(idxs)
            self.ptrs[c]={"idxs":idxs,"p":0,"rng":rng}
    def __len__(self): return self.steps_per_epoch
    def _next_k(self, c, K):
        buf=[]; st=self.ptrs[c]; idxs, p, rng = st["idxs"], st["p"], st["rng"]
        for _ in range(K):
            if p>=len(idxs): rng.shuffle(idxs); p=0
            buf.append(idxs[p]); p+=1
        st["p"]=p; return buf
    def __iter__(self):
        import random
        rng = random.Random(self.seed)
        for _ in range(self.steps_per_epoch):
            chosen = rng.sample(self.classes, self.P) if len(self.classes)>=self.P else [rng.choice(self.classes) for _ in range(self.P)]
            batch=[]
            for c in chosen: batch.extend(self._next_k(c, self.K))
            yield batch

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x): return self.fc(x)

def auc_noise_only_a11_loss(logits, clip_probs, s, alpha=0.3, tau=1.0, eps=1e-12):
    device = logits.device; C = logits.size(1)
    idx = torch.nonzero(s==1, as_tuple=False).squeeze(1)
    m = idx.numel()
    if m < 2: return logits.new_tensor(0.0, requires_grad=True)
    lo = logits[idx]; p_clip = clip_probs[idx].detach()
    p_model = F.softmax(lo, dim=-1)
    q = alpha * p_model + (1.0 - alpha) * p_clip
    offdiag = ~torch.eye(m, dtype=torch.bool, device=device)
    total_loss = lo.new_tensor(0.0); total_w = lo.new_tensor(0.0)
    for i in range(C):
        s_i = lo[:, i]; q_i = q[:, i]
        diff = s_i[:, None] - s_i[None, :]
        w_posneg = (q_i[:, None]) * (1.0 - q_i)[None, :]
        w_negpos = (1.0 - q_i[:, None]) * (q_i)[None, :]
        ell_posneg = F.softplus(-diff / tau); ell_negpos = F.softplus(diff / tau)
        w_sum = (w_posneg + w_negpos)
        loss_mat = w_posneg*ell_posneg + w_negpos*ell_negpos
        total_loss = total_loss + (loss_mat*offdiag).sum()
        total_w    = total_w    + (w_sum   *offdiag).sum()
    return total_loss / (total_w + eps)

@torch.no_grad()
def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048) -> Tuple[float, float]:
    head.eval(); ds = TensorDataset(feats, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ce = nn.CrossEntropyLoss(reduction="mean")
    total_loss=0.0; total_correct=0; total=0
    for xb, yb in dl:
        xb = xb.to(device, dtype=torch.float32); yb = yb.to(device)
        logits = head(xb); loss = ce(logits, yb); pred = logits.argmax(dim=-1)
        total_loss += float(loss.item())*xb.size(0); total_correct += int((pred==yb).sum().item()); total += xb.size(0)
    return total_loss/max(1,total), total_correct/max(1,total)

class EarlyStopping:
    def __init__(self, metric="acc", patience=10, min_delta=0.0):
        assert metric in ("loss","acc"); self.metric=metric
        self.patience=int(patience); self.min_delta=float(min_delta)
        self.best = float("inf") if metric=="loss" else -float("inf"); self.num_bad=0
    def step(self, val_loss: float, val_acc: float) -> bool:
        cur = val_loss if self.metric=="loss" else val_acc
        improved = (cur < self.best - self.min_delta) if self.metric=="loss" else (cur > self.best + self.min_delta)
        if improved: self.best=cur; self.num_bad=0; return False
        self.num_bad += 1; return self.num_bad >= self.patience

def _save_val_curves(epochs, losses, accs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(); plt.plot(epochs, [a*100 for a in accs], marker="o"); plt.xlabel("Epoch"); plt.ylabel("Val Acc (%)")
    plt.title("Validation Accuracy"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_acc.png"), dpi=150); plt.close()
    plt.figure(); plt.plot(epochs, losses, marker="o"); plt.xlabel("Epoch"); plt.ylabel("Val CE Loss")
    plt.title("Validation Loss"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_loss.png"), dpi=150); plt.close()

# --- EMA（可选，更稳的验证） ---
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = LinearHead(model.fc.in_features, model.fc.out_features).to(device)
        self.ema.load_state_dict(model.state_dict()); self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict(); esd = self.ema.state_dict()
        for k in esd.keys():
            esd[k].copy_(esd[k]*self.decay + msd[k]*(1.0-self.decay))

# ----------------------- train -----------------------
def train(
    cache_path="./cache/CIFAR100/clip_ViTL14.pt",
    P=16, K=16, lr=1e-3, lr_max=5e-3, weight_decay=1e-4, max_epochs=200,
    top_outlier_pct=0.30,
    # --- 核心超参（更稳的默认）
    alpha_max=0.3, alpha_warmup_epochs=20, tau=1.0,
    # KD 正则调度
    kd_start=1.0, kd_final=0.2, kd_warmup_epochs=40,
    # 置信门控调度
    conf_start=0.60, conf_final=0.20, conf_warmup_epochs=60,
    # EMA
    use_ema=True, ema_decay=0.999,
    # 监控
    print_every=10,
    ckpt_path="./outputs/CIFAR100/head_auc_noise_only_v2.pt",
    best_ckpt_path="./outputs/CIFAR100/head_auc_noise_only_v2_best.pt",
    early_metric="acc", patience=10, min_delta=0.0,
    val_dir="./val/CIFAR100/auc_noise_only_v2",
):
    set_seed(42)

    # 0) 缓存
    cache = load_clip_cache_strict(cache_path)
    train_feats = F.normalize(cache["train_feats"].to(torch.float32), dim=-1)
    val_feats   = F.normalize(cache["val_feats"].to(torch.float32), dim=-1)
    train_labels= cache["train_labels"].long(); val_labels = cache["val_labels"].long()
    clip_probs_train = cache["clip_probs_train"].to(torch.float32)
    clip_argmax_train= cache["clip_argmax_train"].long()

    N,D = train_feats.shape
    C = int(max(train_labels.max().item(), val_labels.max().item())) + 1
    print(f"[INFO] Ntrain={N}, Nval={val_labels.numel()}, D={D}, C={C} | P={P}, K={K}, batch={P*K}")

    # 1) 得到 s 标记
    y_obs, s_flags = outlier_correction_by_class_centers(train_feats, train_labels, clip_argmax_train, top_pct=top_outlier_pct)

    # 2) 仅不可信
    mask_u = (s_flags == 1)
    if mask_u.sum().item() < P*K: raise RuntimeError("Untrusted subset too small for P×K. Reduce P/K 或降低 top_outlier_pct.")
    feats_u = train_feats[mask_u]; y_true_u = train_labels[mask_u]  # y_true_u 仅日志
    y_obs_u = y_obs[mask_u]; s_u = torch.ones_like(y_obs_u)
    p_clip_u = clip_probs_train[mask_u]
    print(f"[INFO] Using UNTRUSTED ONLY: {feats_u.size(0)} samples.")

    # 3) 数据/采样
    train_ds = FeatureDataset(feats_u, y_true_u, y_obs_u, s_u, p_clip_u)
    batch_sampler = PKBatchSampler(y_obs_u, P=P, K=K, seed=42)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    # 4) 模型
    head = LinearHead(D, C).to(device).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_max, total_steps=max(1, steps_per_epoch*max_epochs),
        pct_start=0.15, anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0
    )
    stopper = EarlyStopping(metric=early_metric, patience=patience, min_delta=min_delta)

    ema = ModelEMA(head, decay=ema_decay) if use_ema else None

    # 5) 记录
    os.makedirs(val_dir, exist_ok=True)
    csv_path = os.path.join(val_dir, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","val_loss","val_acc"])
    val_epochs, val_losses, val_accs = [], [], []

    print("[TRAIN] Start training (UNTRUSTED ONLY + KD + ConfGating) ...")

    for epoch in range(1, max_epochs+1):
        head.train(); t0=time.time()

        # Schedules
        lam_alpha = min(1.0, (epoch-1)/float(alpha_warmup_epochs)) if alpha_warmup_epochs>0 else 1.0
        alpha = alpha_max * lam_alpha

        # KD 权重：从 kd_start -> kd_final（线性/余弦皆可；这里用线性）
        if kd_warmup_epochs > 0:
            t = min(1.0, (epoch-1)/float(kd_warmup_epochs))
            kd_w = kd_start + (kd_final - kd_start) * t
        else:
            kd_w = kd_final

        # 置信门控阈值：从 conf_start -> conf_final 下降
        if conf_warmup_epochs > 0:
            t = min(1.0, (epoch-1)/float(conf_warmup_epochs))
            conf_t = conf_start + (conf_final - conf_start) * t
        else:
            conf_t = conf_final

        running_loss = 0.0; seen = 0

        for ib, batch in enumerate(train_loader, start=1):
            f, _, y_obs_b, s_b, p_clip_b = batch
            f = f.to(device, dtype=torch.float32)
            s_b = s_b.to(device); p_clip_b = p_clip_b.to(device, dtype=torch.float32)

            logits = head(f)
            with torch.no_grad():
                conf = p_clip_b.max(dim=1).values
                mask_conf = (conf >= conf_t)
            # 若门控后样本不足两条，退化为全体（避免空梯度）
            if mask_conf.sum().item() >= 2:
                s_eff = s_b[mask_conf]
                logits_eff = logits[mask_conf]
                p_clip_eff = p_clip_b[mask_conf]
            else:
                s_eff = s_b; logits_eff = logits; p_clip_eff = p_clip_b

            # AUC 主损失
            auc_loss = auc_noise_only_a11_loss(
                logits=logits_eff, clip_probs=p_clip_eff, s=s_eff,
                alpha=alpha, tau=tau
            )

            # KD 稳定器（与 AUC 在相同子集上）
            logp = F.log_softmax(logits_eff, dim=-1)
            kd_loss = -(p_clip_eff * logp).sum(dim=-1).mean()

            loss = auc_loss + kd_w * kd_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step(); scheduler.step()
            if ema is not None: ema.update(head)

            running_loss += float(loss.item()); seen += f.size(0)

            if ib % print_every == 0:
                B = f.size(0); pairs = B*(B-1)
                print(f"[Epoch {epoch:03d}] batch {ib:04d}/{len(train_loader):04d} | "
                      f"seen={seen:6d} | alpha={alpha:.2f} | tau={tau:.2f} | conf_t={conf_t:.2f} "
                      f"| kd_w={kd_w:.2f} | auc={running_loss/ib:.5f} | active={int(mask_conf.sum().item())}/{B}")

        # 验证（EMA 更稳）
        eval_model = ema.ema if ema is not None else head
        val_loss, val_acc = evaluate(eval_model, val_feats, val_labels, batch_size=2048)
        dt = time.time()-t0
        print(f"[Epoch {epoch:03d}] time={dt:.1f}s | avg_train={running_loss/max(1,len(train_loader)):.5f} "
              f"| [VAL] loss={val_loss:.5f} acc={val_acc*100:.2f}%")

        # 记录与图
        val_epochs.append(epoch); val_losses.append(val_loss); val_accs.append(val_acc)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.6f}"])
        _save_val_curves(val_epochs, val_losses, val_accs, val_dir)
        print(f"[VAL-SAVE] CSV -> {csv_path} | PNG -> {val_dir}/val_acc.png, val_loss.png")

        # 保存
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(head.state_dict(), ckpt_path)
        if ema is not None:
            torch.save(ema.ema.state_dict(), ckpt_path.replace(".pt","_ema.pt"))
        # best
        stopper_metric_before = getattr(stopper, "best", None)
        if (early_metric=="loss" and val_loss <= (stopper.best if stopper_metric_before is not None else float("inf"))) or \
           (early_metric=="acc"  and val_acc  >= (stopper.best if stopper_metric_before is not None else -float("inf"))):
            torch.save(eval_model.state_dict(), best_ckpt_path)
            print(f"[CKPT] >>> Best so far ({early_metric}) saved to {best_ckpt_path}")

        if stopper.step(val_loss, val_acc):
            print(f"[EARLY STOP] No improvement in {early_metric} for {stopper.patience} epochs. Stop."); break

    print("[DONE] Training finished.")

# ----------------------- main -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default="./cache/CIFAR100/clip_ViTL14.pt")
    p.add_argument("--P", type=int, default=16); p.add_argument("--K", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--lr_max", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4); p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--top_outlier_pct", type=float, default=0.30)

    p.add_argument("--alpha_max", type=float, default=0.3)
    p.add_argument("--alpha_warmup_epochs", type=int, default=20)
    p.add_argument("--tau", type=float, default=1.0)

    p.add_argument("--kd_start", type=float, default=1.0)
    p.add_argument("--kd_final", type=float, default=0.2)
    p.add_argument("--kd_warmup_epochs", type=int, default=40)

    p.add_argument("--conf_start", type=float, default=0.60)
    p.add_argument("--conf_final", type=float, default=0.20)
    p.add_argument("--conf_warmup_epochs", type=int, default=60)

    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--ckpt", type=str, default="./outputs/CIFAR100/temp.pt")
    p.add_argument("--best_ckpt", type=str, default="./outputs/CIFAR100/head_noise.pt")
    p.add_argument("--early_metric", type=str, default="acc", choices=["loss","acc"])
    p.add_argument("--patience", type=int, default=10); p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_dir", type=str, default="./val/CIFAR100/noise")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args(); set_seed(args.seed)
    train(
        cache_path=args.cache, P=args.P, K=args.K, lr=args.lr, lr_max=args.lr_max, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, top_outlier_pct=args.top_outlier_pct,
        alpha_max=args.alpha_max, alpha_warmup_epochs=args.alpha_warmup_epochs, tau=args.tau,
        kd_start=args.kd_start, kd_final=args.kd_final, kd_warmup_epochs=args.kd_warmup_epochs,
        conf_start=args.conf_start, conf_final=args.conf_final, conf_warmup_epochs=args.conf_warmup_epochs,
        use_ema=args.use_ema, ema_decay=args.ema_decay,
        print_every=args.print_every, ckpt_path=args.ckpt, best_ckpt_path=args.best_ckpt,
        early_metric=args.early_metric, patience=args.patience, min_delta=args.min_delta, val_dir=args.val_dir
    )
