# -*- coding: utf-8 -*-
"""
train_hcl_loss_cifar100.py

要求实现：
- 保留你的“类内中心 + 余弦距离筛每类 top-30% 离群样本 -> 人工校正（用真值替换 y），并标 s=0，其余 s=1”
- 取消 P×K 采样，使用标准 DataLoader(shuffle=True)
- 使用论文的风险一致估计 + MSE-like 类损失（Eq.6, Eq.7, Eq.8），并且 **不对 P_model 断梯度**
- 训练细节尽量与论文一致：CLIP ViT-L/14 冻结 + 线性头、30 epochs、batch=64、AdamW(5e-4,1e-4)、StepLR(5,0.1)、默认 λ=1.0

公式出处（HCL）：
Eq.6（风险重写）、Eq.7（MSE-like）、Eq.8（经验近似）、Eq.9-11（P_model, P_CLIP, 插值）



实现要点：
- 我们的数据划分语义与论文 s 的语义不同：你的 s=0 表示“已人工校正（可信，硬标签）”；s=1 表示“未校正（用条件分布软监督）”。
- 经验风险：
  R_hat = (1/|D_H|) * Σ_{x∈D_H} L(f(x), y) + (1/|D_V|) * Σ_{x∈D_V} Σ_i P_hat(i|Y,s=0,x) * L(f(x), i)
- 条件分布估计（Eq.11）：P_hat = λ * P_CLIP + (1-λ) * P_model；且 **P_model 不做 detach**（按你的要求）。

使用：
python src/contrast/train_hcl_CIFAR100.py
"""

import os, math, pickle, json, random, time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 你的 CLIP 实现
from clip import clip
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import clip_cache, dataset_root, experiment_checkpoint, save_experiment_config

# -----------------------
# 全局配置
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_PATH = str(clip_cache("CIFAR100"))  # 预计算缓存（float32）
CACHE_DIR = os.path.dirname(CACHE_PATH) or "."

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# -----------------------
# CIFAR-100 读取
# -----------------------
def load_cifar100():
    with open(dataset_root("CIFAR100") / "train", 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')
    with open(dataset_root("CIFAR100") / "test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')
    with open(dataset_root("CIFAR100") / "meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')
    return data_train, data_test, data_meta

def read_data_cifar_100():
    data_train, data_test, data_meta = load_cifar100()
    train_data = data_train['data'].reshape((data_train['data'].shape[0], 3, 32, 32))
    test_data  = data_test['data'].reshape((data_test['data'].shape[0],  3, 32, 32))
    train_label = np.array(data_train["fine_labels"], dtype=np.int64)
    test_label  = np.array(data_test["fine_labels"], dtype=np.int64)
    fine_label_names = data_meta['fine_label_names']  # 100 类名
    return train_data, train_label, test_data, test_label, fine_label_names

# -----------------------
# CLIP & 预处理
# -----------------------
def get_transform(image_size=224):
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    def convert_to_rgb(img): return img.convert("RGB")
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def load_clip_model():
    model, _ = clip.load("ViT-L/14", device=DEVICE, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # 冻结（与论文一致）
    return model

def build_text_embeddings(model, classnames):
    # 与你原实现一致（简单模板）
    single_template = ["a photo of a {}."]
    def processed_name(name):
        return name.replace("_", " ").replace("/", " or ").lower().rstrip(".")
    texts = []
    for cname in classnames:
        prompts = [tmpl.format(processed_name(cname)) for tmpl in single_template]
        prompts = ["This is " + t if (t.startswith("a") or t.startswith("the")) else t for t in prompts]
        texts.append(prompts)
    with torch.no_grad():
        all_embeds = []
        for prompts in texts:
            tokens = clip.tokenize(prompts).to(DEVICE)
            txt = model.encode_text(tokens)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            txt = txt.mean(dim=0)
            txt = txt / txt.norm()
            all_embeds.append(txt)
        label_embed = torch.stack(all_embeds, dim=0)  # [C, d]
    return label_embed.float()

@torch.no_grad()
def encode_split_images(model, imgs_np, transform, batch=256):
    N = imgs_np.shape[0]
    feats = []
    for i in tqdm(range(0, N, batch), desc="Encode CLIP image features"):
        bs = min(batch, N - i)
        batch_imgs = []
        for j in range(bs):
            img = Image.fromarray(np.uint8(imgs_np[i+j]).transpose(1,2,0))
            batch_imgs.append(transform(img))
        x = torch.stack(batch_imgs, dim=0).to(DEVICE)
        f = model.encode_image(x)  # [bs, d]
        f = f / f.norm(dim=-1, keepdim=True)  # 单位化
        feats.append(f)
    feats = torch.cat(feats, dim=0)
    return feats

@torch.no_grad()
def compute_clip_probs(model, img_feats, text_embeds):
    # 论文 Eq.10 使用 τ·cos 后 softmax；这里用 CLIP 自带 logit_scale≈τ（clamp 到 100）
    logit_scale = model.logit_scale.exp().float().clamp(max=100.0)
    text_embeds = text_embeds.to(dtype=img_feats.dtype)
    logits = logit_scale * (img_feats @ text_embeds.t())  # [N, C]
    probs = F.softmax(logits.float(), dim=-1)
    preds = probs.argmax(dim=-1)
    return probs, preds, logits

def build_or_load_cache(image_size=224) -> Dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH):
        print(f"[cache] loading: {CACHE_PATH}")
        cache = torch.load(CACHE_PATH, map_location="cpu")
        for k in ["train_feats","test_feats","clip_probs_train","clip_probs_test"]:
            cache[k] = cache[k].float()
        return cache

    print("[cache] not found. Building CLIP features and zero-shot probs ...")
    model = load_clip_model()
    transform = get_transform(image_size=image_size)

    trX, trY, teX, teY, classnames = read_data_cifar_100()
    trY_t = torch.from_numpy(trY)
    teY_t = torch.from_numpy(teY)

    text_embeds = build_text_embeddings(model, classnames).to(DEVICE)
    train_feats = encode_split_images(model, trX, transform).float()
    test_feats  = encode_split_images(model, teX, transform).float()

    clip_probs_train, clip_preds_train, _ = compute_clip_probs(model, train_feats, text_embeds)
    clip_probs_test,  clip_preds_test,  _ = compute_clip_probs(model, test_feats,  text_embeds)

    cache = {
        "classnames": classnames,
        "train_labels": trY_t,
        "test_labels":  teY_t,
        "train_feats":  train_feats.cpu(),
        "test_feats":   test_feats.cpu(),
        "text_embeds":  text_embeds.cpu(),
        "clip_probs_train": clip_probs_train.cpu(),
        "clip_probs_test":  clip_probs_test.cpu(),
        "clip_preds_train": clip_preds_train.cpu(),
        "clip_preds_test":  clip_preds_test.cpu(),
    }
    torch.save(cache, CACHE_PATH)
    print(f"[cache] saved to: {CACHE_PATH}")
    return cache

# -----------------------
# 你的“人工校正”筛选：类内中心 + 余弦距离 top-30% 离群
# -----------------------
def per_class_outliers_by_cos(train_feats: torch.Tensor,
                              clip_pred: torch.Tensor,
                              top_ratio: float = 0.3):
    """
    以 CLIP 预测标签为类划分，按与类中心的余弦距离(=1-cos)降序取 top_ratio 作为离群索引。
    返回：outlier_mask(bool)[N]，每类中心 centers[C,d]
    """
    N, d = train_feats.shape
    C = int(clip_pred.max().item()) + 1
    centers = torch.zeros(C, d)
    outlier_mask = torch.zeros(N, dtype=torch.bool)

    for c in range(C):
        idx = (clip_pred == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            centers[c] = torch.zeros(d)
            continue
        f = train_feats[idx]
        center = f.mean(dim=0)
        center = center / (center.norm() + 1e-12)
        centers[c] = center
        cos_sim = (f @ center)
        cos_dist = 1.0 - cos_sim.clamp(-1, 1)  # 越大越离群
        k = max(1, int(math.ceil(top_ratio * idx.numel())))
        topk = torch.topk(cos_dist, k=k, largest=True).indices
        outlier_mask[idx[topk]] = True

    return outlier_mask, centers

def build_s_and_observed_labels(train_true: torch.Tensor,
                                clip_pred: torch.Tensor,
                                outlier_mask: torch.Tensor):
    """
    仅对离群子集做“人工校正=用真值替换”，并记为 s=0；其余样本 s=1。
    注意：这与论文 s 的语义相反，但本实现按你的语义用于经验风险两部分。
    """
    y_obs = clip_pred.clone()
    y_obs[outlier_mask] = train_true[outlier_mask]
    s = torch.ones_like(train_true, dtype=torch.long)  # 1=未校正（不可信）
    s[outlier_mask] = 0                                # 0=已校正（可信）
    return s, y_obs

# -----------------------
# Dataset / DataLoader（不使用 P×K）
# -----------------------
class FeatureDataset(Dataset):
    def __init__(self, feats, labels, s, p_clip, y_obs):
        self.feats  = feats
        self.labels = labels
        self.s      = s
        self.p_clip = p_clip
        self.y_obs  = y_obs
    def __len__(self): return self.feats.size(0)
    def __getitem__(self, idx):
        return (self.feats[idx], self.labels[idx], self.s[idx],
                self.p_clip[idx], self.y_obs[idx])

# -----------------------
# 线性头（与论文一致：CLIP 冻结 + 线性层）
# -----------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=False)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
    def forward(self, x):
        return self.fc(x)  # logits（margin）

# -----------------------
# HCL 的 MSE-like 类损失（Eq.7）+ 经验风险（Eq.8）
# -----------------------
@torch.no_grad()
def _class_codebook(C: int, device):
    """
    生成 T[i, j] = +1 (i==j) else -1 的 [C, C] 码本，便于一次性计算 L(f(x), i)。
    """
    T = -torch.ones(C, C, device=device, dtype=torch.float32)
    T.fill_(-1.0)
    T[torch.arange(C), torch.arange(C)] = 1.0
    return T

def hcl_empirical_risk(
    logits: torch.Tensor,     # [B,C]
    y_obs: torch.Tensor,      # [B]
    s: torch.Tensor,          # [B] 0=已校正（硬监督），1=未校正（软监督）
    p_clip: torch.Tensor,     # [B,C]
    lambda_clip: float = 1.0  # Eq.11, 默认 1.0（只用 CLIP）
) -> torch.Tensor:
    """
    不对 P_model 断梯度（按你的要求）。
    """
    B, C = logits.shape
    device = logits.device

    # 计算 per-class MSE-like L(f(x), i)（Eq.7） -> [B,C]
    T = _class_codebook(C, device=device)          # [C,C]
    diff = 1.0 - logits.unsqueeze(1) * T.unsqueeze(0)  # [B,C,C]
    per_class_loss = (diff ** 2).mean(dim=-1)          # [B,C]

    # Eq.9：P_model（允许梯度回传）
    p_model = F.softmax(logits, dim=-1)  # [B,C]

    # Eq.11：P_hat 条件分布（允许梯度通过 p_model）
    p_hat = lambda_clip * p_clip + (1.0 - lambda_clip) * p_model  # [B,C]

    # 经验风险两部分（Eq.8）
    mask_corr = (s == 0)  # 已校正：硬标签
    mask_unc  = (s == 1)  # 未校正：条件分布期望

    total_terms = 0
    total = 0.0

    if mask_corr.any():
        y_corr = y_obs[mask_corr]
        hard = per_class_loss[mask_corr, :][torch.arange(y_corr.numel(), device=device), y_corr]  # [Nc]
        total = total + hard.mean()
        total_terms += 1

    if mask_unc.any():
        soft = (per_class_loss[mask_unc, :] * p_hat[mask_unc, :]).sum(dim=-1)  # [Nu]
        total = total + soft.mean()
        total_terms += 1

    if total_terms == 0:
        return per_class_loss.mean()  # 兜底

    return total

# -----------------------
# 评估
# -----------------------
@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item() * 100.0

@torch.no_grad()
def eval_linear_head(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch=2048) -> float:
    head.eval()
    outs = []
    for i in range(0, feats.size(0), batch):
        x = feats[i:i+batch].to(DEVICE)
        outs.append(head(x).float().cpu())
    logits = torch.cat(outs, dim=0)
    return accuracy_top1(logits, labels)

@torch.no_grad()
def eval_clip_zeroshot(model, feats: torch.Tensor, text_embeds: torch.Tensor, labels: torch.Tensor, batch=4096) -> float:
    logit_scale = model.logit_scale.exp().float().clamp(max=100.0)
    acc_sum, n = 0.0, 0
    for i in range(0, feats.size(0), batch):
        f = feats[i:i+batch].to(DEVICE)
        te = text_embeds.to(DEVICE, dtype=f.dtype)
        logits = logit_scale * (f @ te.t())
        acc_sum += accuracy_top1(logits.float().cpu(), labels[i:i+batch]) * f.size(0)
        n += f.size(0)
    return acc_sum / max(1, n)

# -----------------------
# 训练主循环（论文默认设置）
# -----------------------
@dataclass
class TrainConfig:
    image_size: int = 224
    batch_size: int = 64              # 与论文一致
    epochs: int = 30                  # 与论文一致
    lr: float = 5e-4                  # AdamW
    weight_decay: float = 1e-4
    lambda_clip: float = 1.0          # Eq.11 缺省 1.0（论文默认）
    patience: int = 10                # 可选：若你想固定 30 epoch，可忽略早停
    log_interval: int = 50

def train():
    cfg = TrainConfig()
    print(cfg)

    cache = build_or_load_cache(image_size=cfg.image_size)

    classnames  = cache["classnames"]
    train_labels= cache["train_labels"].long()
    test_labels = cache["test_labels"].long()
    train_feats = cache["train_feats"].float()
    test_feats  = cache["test_feats"].float()
    text_embeds = cache["text_embeds"].float()
    p_clip_train= cache["clip_probs_train"].float()

    C = len(classnames)
    d = train_feats.shape[1]

    # 你的“人工校正”选择（保留）
    outlier_mask, _ = per_class_outliers_by_cos(train_feats, cache["clip_preds_train"].long(), top_ratio=0.3)
    s_train, y_obs = build_s_and_observed_labels(train_labels, cache["clip_preds_train"].long(), outlier_mask)

    # 日志：zero-shot
    model_clip = load_clip_model()
    zs_train_acc = eval_clip_zeroshot(model_clip, train_feats, text_embeds, train_labels)
    zs_test_acc  = eval_clip_zeroshot(model_clip, test_feats,  text_embeds, test_labels)
    print(f"[zero-shot] train: {zs_train_acc:.2f}% | test: {zs_test_acc:.2f}%")

    # DataLoader（不再使用 P×K）
    ds = FeatureDataset(train_feats, train_labels, s_train, p_clip_train, y_obs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # 线性头 + 优化器/调度（论文设置）
    head = LinearHead(in_dim=d, num_classes=C).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched= torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    best_acc = -1.0
    best_state = None
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs+1):
        head.train()
        running = 0.0
        t0 = time.time()

        for step, batch in enumerate(dl, start=1):
            x, y, s, pc, yob = batch
            x   = x.to(DEVICE)
            y   = y.to(DEVICE)
            s   = s.to(DEVICE)
            pc  = pc.to(DEVICE)
            yob = yob.to(DEVICE)

            logits = head(x)  # [B,C]
            loss = hcl_empirical_risk(
                logits=logits,
                y_obs=yob,
                s=s,
                p_clip=pc,
                lambda_clip=cfg.lambda_clip  # 默认 1.0（只用 CLIP）
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            opt.step()

            running += loss.item()
            if step % cfg.log_interval == 0:
                print(f"[epoch {epoch:02d} step {step:04d}] loss={running/step:.4f}")

        sched.step()

        # 评估 top-1
        train_acc = eval_linear_head(head, train_feats, train_labels)
        val_acc   = eval_linear_head(head,  test_feats,  test_labels)
        dt = time.time() - t0
        print(f"[epoch {epoch:02d}] loss={running/max(1,len(dl)):.4f} | train@1={train_acc:.2f}% | "
              f"val@1={val_acc:.2f}% | lr={sched.get_last_lr()[0]:.4e} | {dt:.1f}s")

        # 可选早停：若你想完全与论文一致，可忽略早停逻辑
        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in head.state_dict().items()}
            experiment_checkpoint("CIFAR100_hcl", "best.ckpt").parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, experiment_checkpoint("CIFAR100_hcl", "best.ckpt"))
            patience_left = cfg.patience
            print(f"  ↳ new best! val@1={best_acc:.2f}% (saved {experiment_checkpoint('CIFAR100_hcl', 'best.ckpt')})")
        else:
            patience_left -= 1
            print(f"  ↳ no improve. patience_left={patience_left}")
            if patience_left <= 0:
                print("  ↳ early stopped.")
                break

    print("="*60)
    print(f"[final] zero-shot train: {zs_train_acc:.2f}% | zero-shot test: {zs_test_acc:.2f}%")
    if best_state is not None:
        head.load_state_dict(best_state)
    final_train_acc = eval_linear_head(head, train_feats, train_labels)
    final_test_acc  = eval_linear_head(head,  test_feats,  test_labels)
    print(f"[final] linear-head train: {final_train_acc:.2f}% | linear-head test: {final_test_acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    save_experiment_config("CIFAR100_hcl", {"cache": CACHE_PATH, "seed": RNG_SEED})
    train()
