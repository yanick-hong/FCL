# -*- coding: utf-8 -*-
"""
train_DLD_CIFAR100.py  —— DLD-Linear + 并行线性头（eval_acc 友好，val_dir 可配置）

与先前版本相比：
- 不做离群人工校正；从 make_obs_labels.py 读取 y_obs/s
- 新增并行 LinearHead，联合训练：
    L = L_DLD + head_lambda * CE(linear(f), y_obs)
                + 0.5*head_lambda * KL( softmax(linear(f)/T) || softmax(stopgrad(y0_hat)/T) )
  其中 y0_hat = y_t - a*yd_hat - b*eps_hat
- 验证/早停以“线性头”的 CE/Top1 为准；仅保存 best_ckpt
- 保存的 state_dict 同时含 DLD 参数与 linear.weight/bias，兼容 eval_acc.py
"""

import os, math, csv, time, argparse
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- 通用 ---------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_clip_cache_strict(cache_path: str) -> Dict[str, torch.Tensor]:
    assert os.path.isfile(cache_path), f"Cache not found: {cache_path}"
    data = torch.load(cache_path, map_location="cpu")
    fp32 = ["train_feats","val_feats","test_feats","clip_probs_train","clip_probs_val","clip_probs_test","clip_label_embeds"]
    lng  = ["train_labels","val_labels","test_labels","clip_argmax_train","clip_argmax_val","clip_argmax_test","train_idx_from_train50k","val_idx_from_train50k"]
    for k in fp32:
        if k in data and isinstance(data[k], torch.Tensor): data[k] = data[k].to(torch.float32)
    for k in lng:
        if k in data and isinstance(data[k], torch.Tensor): data[k] = data[k].long()
    return data

def load_obs_labels(obs_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(obs_path): raise FileNotFoundError(obs_path)
    data = torch.load(obs_path, map_location="cpu")
    if not ("y_obs" in data and "s" in data): raise KeyError(f"need y_obs & s, got {list(data.keys())}")
    return {"y_obs": data["y_obs"].long(), "s": data["s"].long()}

# --------------- 数据与采样 ---------------
class FeatureDataset(Dataset):
    def __init__(self, feats, y_true, y_obs, p_clip):
        self.f, self.y_true, self.y_obs, self.p_clip = feats, y_true, y_obs, p_clip
    def __len__(self): return self.f.size(0)
    def __getitem__(self, i): return self.f[i], int(self.y_true[i]), int(self.y_obs[i]), self.p_clip[i]

class PKBatchSampler(Sampler[List[int]]):
    def __init__(self, labels_obs: torch.Tensor, P: int, K: int, seed: int = 42):
        super().__init__(None)
        import random
        lab = labels_obs.cpu().numpy()
        self.P, self.K, self.seed = int(P), int(K), int(seed)
        self.class_to_indices = {int(c): np.where(lab==c)[0].tolist() for c in np.unique(lab)}
        self.classes = list(self.class_to_indices.keys())
        self.N = len(lab); self.batch_size = self.P*self.K
        self.steps_per_epoch = max(1, self.N//self.batch_size)
        self.ptrs = {}
        for c, idxs in self.class_to_indices.items():
            rng = random.Random(self.seed+c); rng.shuffle(idxs)
            self.ptrs[c] = {"idxs": idxs, "p": 0, "rng": rng}
    def __len__(self): return self.steps_per_epoch
    def _next_k(self, c, K):
        st = self.ptrs[c]; idxs, p, rng = st["idxs"], st["p"], st["rng"]; out=[]
        for _ in range(K):
            if p>=len(idxs): rng.shuffle(idxs); p=0
            out.append(idxs[p]); p+=1
        st["p"]=p; return out
    def __iter__(self):
        import random
        rng=random.Random(self.seed)
        for _ in range(self.steps_per_epoch):
            chosen = rng.sample(self.classes, self.P) if len(self.classes)>=self.P else [rng.choice(self.classes) for _ in range(self.P)]
            batch=[]
            for c in chosen: batch.extend(self._next_k(c, self.K))
            yield batch

# --------------- 模型 ---------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64): super().__init__(); self.dim=dim
    def forward(self, t):
        dev=t.device; half=self.dim//2
        freqs=torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=dev))
        ang=t[:,None]*freqs[None,:]*2*math.pi
        emb=torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return F.pad(emb,(0,1),value=0.0) if self.dim%2==1 else emb

class DLDLinear(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, tdim=64):
        super().__init__(); C=num_classes; H=hidden
        self.t_embed=SinusoidalTimeEmbedding(tdim)
        self.proj_f=nn.Sequential(nn.Linear(in_dim,H), nn.GELU(), nn.LayerNorm(H))
        self.proj_y=nn.Sequential(nn.Linear(2*C,H),   nn.GELU(), nn.LayerNorm(H))
        self.proj_t=nn.Sequential(nn.Linear(tdim,H),  nn.GELU(), nn.LayerNorm(H))
        self.head  =nn.Sequential(nn.Linear(H,H),     nn.GELU(), nn.LayerNorm(H))
        self.out_d   = nn.Linear(H, C); nn.init.xavier_uniform_(self.out_d.weight); nn.init.zeros_(self.out_d.bias)
        self.out_eps = nn.Linear(H, C); nn.init.xavier_uniform_(self.out_eps.weight); nn.init.zeros_(self.out_eps.bias)
    def forward(self, x, y_t, y_n, t):
        h = self.proj_f(x) + self.proj_y(torch.cat([y_t,y_n],dim=-1)) + self.proj_t(self.t_embed(t))
        h = self.head(h)
        return self.out_d(h), self.out_eps(h)

class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)

# --------------- DLD 工具 ---------------
def a_func(t): return t
def b_func(t): return t
def one_hot(y, C): return F.one_hot(y.clamp_min(0), num_classes=C).to(torch.float32)
def smooth_probs(p, tau=0.7, eps=1e-6):
    p=p.float().clamp_min(eps); return F.softmax(torch.log(p)/max(1e-6,tau), dim=-1)

@torch.no_grad()
def dld_predict_proba(model, feats, p_clip, steps=5, batch_size=2048):
    model.eval(); probs_out=[]
    dl=DataLoader(TensorDataset(feats, p_clip), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
    t_grid=torch.linspace(1.0,0.0,steps=steps+1, device=device)
    for xb,pb in dl:
        xb=xb.to(device).float(); y_n=smooth_probs(pb.to(device).float())
        y_t=y_n.clone()
        for k in range(steps,0,-1):
            t_k=torch.full((xb.size(0),), float(t_grid[k].item()), device=device)
            t_km1=torch.full((xb.size(0),), float(t_grid[k-1].item()), device=device)
            a_k=a_func(t_k); b_k=b_func(t_k)
            yd_hat, eps_hat = model(xb, y_t, y_n, t_k)
            y0_hat = y_t - a_k[:,None]*yd_hat - b_k[:,None]*eps_hat
            y_t = y0_hat + a_func(t_km1)[:,None]*yd_hat
        probs = F.softmax(y0_hat, dim=-1)
        probs_out.append(probs.detach().cpu())
    return torch.cat(probs_out,0)

@torch.no_grad()
def evaluate_dld(model, feats, labels, p_clip, steps=5, batch_size=2048):
    probs=dld_predict_proba(model, feats, p_clip, steps=steps, batch_size=batch_size).to(device)
    labels=labels.to(device)
    ce=F.nll_loss((probs+1e-12).log(), labels, reduction="mean").item()
    acc=(probs.argmax(dim=-1)==labels).float().mean().item()
    return ce, acc

@torch.no_grad()
def evaluate_head(head, feats, labels, batch_size=2048):
    head.eval(); ce_sum=0.0; correct=0; total=0
    dl=DataLoader(TensorDataset(feats, labels), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
    for xb,yb in dl:
        xb=xb.to(device).float(); yb=yb.to(device).long()
        logits=head(xb)
        ce_sum+=F.cross_entropy(logits, yb, reduction="sum").item()
        correct+=(logits.argmax(-1)==yb).sum().item()
        total+=yb.numel()
    return ce_sum/max(1,total), correct/max(1,total)

class EarlyStopping:
    def __init__(self, metric="acc", patience=10, min_delta=0.0):
        assert metric in ("loss","acc"); self.metric=metric; self.patience=int(patience); self.min_delta=float(min_delta)
        self.best = float("inf") if metric=="loss" else -float("inf"); self.num_bad=0
    def step(self, val_loss, val_acc):
        cur = val_loss if self.metric=="loss" else val_acc
        improved = (cur < self.best - self.min_delta) if self.metric=="loss" else (cur > self.best + self.min_delta)
        if improved: self.best=cur; self.num_bad=0; return False
        self.num_bad+=1; return self.num_bad>=self.patience

def _save_val_curves(epochs, losses, accs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(); plt.plot(epochs, [a*100 for a in accs], marker="o"); plt.xlabel("Epoch"); plt.ylabel("Val Acc (%)")
    plt.title("Validation Accuracy (Linear Head)"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"val_acc.png"), dpi=150); plt.close()
    plt.figure(); plt.plot(epochs, losses, marker="o"); plt.xlabel("Epoch"); plt.ylabel("Val CE Loss")
    plt.title("Validation Loss (Linear Head)"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"val_loss.png"), dpi=150); plt.close()

# --------------- 训练主流程 ---------------
def train(
    cache_path=str(clip_cache("CIFAR100")),
    obs_labels_path=str(observed_labels_cache("CIFAR100")),
    P=16, K=16, lr=1e-3, lr_max=5e-3, weight_decay=1e-4, max_epochs=200,
    hidden=512, tdim=64, steps=5, tau_clip=0.7,
    print_every=20, best_ckpt_path=str(experiment_checkpoint("CIFAR100_dld")),
    early_metric="acc", patience=10, min_delta=0.0, val_dir=str(experiment_logs("CIFAR100_dld")),
    # 新增：线性头联合训练的强度（一个 knob），以及对齐温度
    head_lambda: float = 0.2, align_T: float = 2.0, seed: int = 42,
):
    set_seed(seed)

    cache=load_clip_cache_strict(cache_path)
    train_feats=F.normalize(cache["train_feats"].to(torch.float32), dim=-1)
    val_feats  =F.normalize(cache["val_feats"].to(torch.float32),   dim=-1)
    train_labels=cache["train_labels"].long(); val_labels=cache["val_labels"].long()
    p_train=cache["clip_probs_train"].to(torch.float32); p_val=cache["clip_probs_val"].to(torch.float32)

    N,D=train_feats.shape; C=int(max(train_labels.max().item(), val_labels.max().item()))+1
    obs = load_obs_labels(obs_labels_path); y_obs_all, s_flags = obs["y_obs"], obs["s"]
    if y_obs_all.numel()!=N: raise RuntimeError("obs length mismatch")

    train_ds=FeatureDataset(train_feats, train_labels, y_obs_all, p_train)
    batch_sampler=PKBatchSampler(labels_obs=y_obs_all, P=P, K=K, seed=42)
    train_loader=DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    dld = DLDLinear(in_dim=D, num_classes=C, hidden=hidden, tdim=tdim).to(device)
    head= LinearHead(in_dim=D, num_classes=C).to(device)
    optimizer=torch.optim.AdamW(list(dld.parameters())+list(head.parameters()), lr=lr, weight_decay=weight_decay)
    steps_per_epoch=len(train_loader)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr_max, total_steps=max(1, steps_per_epoch*max_epochs),
                                                  pct_start=0.15, anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0)
    stopper=EarlyStopping(metric=early_metric, patience=patience, min_delta=min_delta)

    os.makedirs(val_dir, exist_ok=True)
    csv_path=os.path.join(val_dir,"val_metrics_dld_linear.csv")
    with open(csv_path,"w",newline="") as f: csv.writer(f).writerow(["epoch","val_ce_head","val_acc_head"])
    val_epochs,val_losses,val_accs=[],[],[]

    mse=nn.MSELoss(reduction="mean")

    best_sd=None  # 合并后的最优 state_dict（含 linear.*）
    best_metric = -float("inf") if early_metric=="acc" else float("inf")

    print("[TRAIN] Start DLD + LinearHead joint training ...")
    for epoch in range(1, max_epochs+1):
        dld.train(); head.train(); run_loss=0.0; t0=time.time()

        for ib, (f, y_true, y_obs, p_clip) in enumerate(train_loader, start=1):
            f=f.to(device).float(); y_obs=y_obs.to(device).long(); p_clip=p_clip.to(device).float()
            B=f.size(0)
            y0=one_hot(y_obs, C); y_n=smooth_probs(p_clip, tau=tau_clip); y_d=(y_n - y0)
            t=torch.rand(B, device=device); a_t=a_func(t); b_t=b_func(t); eps=torch.randn_like(y0)
            y_t = y0 + a_t[:,None]*y_d + b_t[:,None]*eps

            yd_hat, eps_hat = dld(f, y_t, y_n, t)
            y0_hat = y_t - a_t[:,None]*yd_hat - b_t[:,None]*eps_hat  # DLD 的一步复原

            logits = head(f)

            loss_dld = mse(yd_hat, y_d) + mse(eps_hat, eps)
            loss_head = F.cross_entropy(logits, y_obs)

            # 对齐蒸馏（不反传到 DLD）：KL( softmax(logits/T) || softmax(stopgrad(y0_hat)/T) )
            with torch.no_grad():
                target_soft = F.softmax(y0_hat / align_T, dim=-1)
            logp = F.log_softmax(logits / align_T, dim=-1)
            loss_align = F.kl_div(logp, target_soft, reduction="batchmean") * (align_T*align_T)  # 常见做法：乘 T^2

            loss = loss_dld + head_lambda * loss_head + 0.5*head_lambda * loss_align

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(dld.parameters())+list(head.parameters()), max_norm=5.0)
            optimizer.step(); scheduler.step()
            run_loss += float(loss.item())

            if ib % print_every == 0:
                print(f"[Epoch {epoch:03d}] batch {ib:04d}/{len(train_loader):04d} | "
                      f"lr={scheduler.get_last_lr()[0]:.3e} | "
                      f"Ldld={loss_dld.item():.4f} Lhead={loss_head.item():.4f} Lalign={loss_align.item():.4f} | "
                      f"Loss(avg)={run_loss/ib:.5f}")

        # 验证：线性头（用于早停与保存）
        val_ce_head, val_acc_head = evaluate_head(head, val_feats, val_labels, batch_size=2048)
        # 同时打印 DLD 的指标供参考（不作为早停）
        val_ce_dld,  val_acc_dld  = evaluate_dld(dld,  val_feats, val_labels, p_val, steps=steps, batch_size=2048)
        print(f"[VAL] epoch={epoch:03d} | Head: CE={val_ce_head:.4f} Acc={val_acc_head*100:.2f}% "
              f"| DLD: CE={val_ce_dld:.4f} Acc={val_acc_dld*100:.2f}% | time={time.time()-t0:.1f}s")

        val_epochs.append(epoch); val_losses.append(val_ce_head); val_accs.append(val_acc_head)
        with open(csv_path,"a",newline="") as f: csv.writer(f).writerow([epoch, f"{val_ce_head:.6f}", f"{val_acc_head:.6f}"])
        _save_val_curves(val_epochs, val_losses, val_accs, val_dir)

        # 仅按线性头指标判定最优并保存（兼容 eval_acc.py）
        improved = (val_acc_head > best_metric) if early_metric=="acc" else (val_ce_head < best_metric)
        if improved:
            best_metric = val_acc_head if early_metric=="acc" else val_ce_head
            merged = dict(dld.state_dict())
            merged["linear.weight"] = head.linear.weight.detach().cpu()
            merged["linear.bias"]   = head.linear.bias.detach().cpu()
            best_sd = merged
            if best_ckpt_path:
                payload = {
                    "state_dict": best_sd,
                    "in_dim": D, "num_classes": C,
                    "hidden": hidden, "tdim": tdim,
                    "note": "DLD params + linear.* for eval_acc.py",
                }
                os.makedirs(os.path.dirname(best_ckpt_path) or ".", exist_ok=True)
                torch.save(payload, best_ckpt_path)
                print(f"[CKPT] >>> Saved best linear head to {best_ckpt_path}")

        if stopper.step(val_ce_head if early_metric=="loss" else (1.0 - val_acc_head),  # 仅用于计数；已在 improved 中保存
                        val_acc_head):
            print(f"[EARLY STOP] No improvement in {early_metric} for {stopper.patience} epochs. Stop."); break

    print("[DONE] Joint training finished.")

# --------------- CLI ---------------
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--cache", type=str, default=str(clip_cache("CIFAR100")))
    p.add_argument("--obs_labels_path", type=str, default=str(observed_labels_cache("CIFAR100")))
    p.add_argument("--P", type=int, default=32); p.add_argument("--K", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--lr_max", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4); p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=512); p.add_argument("--tdim", type=int, default=64)
    p.add_argument("--steps", type=int, default=5); p.add_argument("--tau_clip", type=float, default=0.7)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--best_ckpt", type=str, default=str(experiment_checkpoint("CIFAR100_dld")))
    p.add_argument("--early_metric", type=str, default="acc", choices=["loss","acc"])
    p.add_argument("--patience", type=int, default=10); p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--val_dir", type=str, default=str(experiment_logs("CIFAR100_dld")))
    p.add_argument("--seed", type=int, default=42)
    # 一个超参：线性头强度；以及对齐温度（通常 2~4）
    p.add_argument("--head_lambda", type=float, default=0.2)
    p.add_argument("--align_T", type=float, default=2.0)
    return p.parse_args()

if __name__ == "__main__":
    args=parse_args(); save_experiment_config("CIFAR100_dld", args); set_seed(args.seed)
    train(cache_path=args.cache, obs_labels_path=args.obs_labels_path,
          P=args.P, K=args.K, lr=args.lr, lr_max=args.lr_max, weight_decay=args.weight_decay, max_epochs=args.max_epochs,
          hidden=args.hidden, tdim=args.tdim, steps=args.steps, tau_clip=args.tau_clip,
          print_every=args.print_every, best_ckpt_path=args.best_ckpt,
          early_metric=args.early_metric, patience=args.patience, min_delta=args.min_delta, val_dir=args.val_dir,
          head_lambda=args.head_lambda, align_T=args.align_T, seed=args.seed)
