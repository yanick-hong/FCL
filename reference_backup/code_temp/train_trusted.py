# -*- coding: utf-8 -*-
"""
train_trusted.py

（Trusted-only）线性头训练：
- 读取缓存,读取 make_obs_labels.py 的输出（含 y_obs 与 s），仅用 s==0 的可信子集训练
- 仅训练线性头（CLIP 已冻结于缓存阶段）
- 评估：全训练集 / 可信子集 / 测试集 的精度与 CE
- 仅保存最佳 ckpt（无临时保存）
"""

import os, time, copy, pickle, argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision.transforms import Normalize, Compose, Resize, ToTensor

# 如果你机器上安装了 openai-clip 包（clip），下方两行才能在首次构建缓存时工作
# 但通常我们只“读取”现有缓存，不需要真正构建
try:
    from clip import clip  # pip install git+https://github.com/openai/CLIP.git
except Exception:
    clip = None

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# 通用工具
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def convert_to_rgb(image): return image.convert("RGB")

def get_transform(image_size=224):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def article(name): return "an" if name[0].lower() in "aeiou" else "a"

def processed_name(name, rm_dot=False):
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot: res = res.rstrip(".")
    return res

def load_cifar100():
    with open("./datasets/CIFAR100/train", 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')
    with open("./datasets/CIFAR100/test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')
    with open("./datasets/CIFAR100/meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')
    return data_train, data_test, data_meta

def read_data_cifar_100():
    data_train, data_test, data_meta = load_cifar100()
    train_data = data_train['data'].reshape((data_train['data'].shape[0], 3, 32, 32))
    test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))
    train_label = data_train["fine_labels"]
    test_label = data_test["fine_labels"]
    return train_data, train_label, test_data, test_label

def load_taglist(dataset: str = "CIFAR100") -> Dict[str, List[str]]:
    tag_file = f"./datasets/{dataset}/{dataset}_ram_taglist.txt"
    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]
    return {"taglist": taglist_or}

single_template = ["a photo of a {}."]

@torch.no_grad()
def build_clip_label_embedding(model, categories: List[str]) -> torch.Tensor:
    templates = single_template
    run_on_gpu = torch.cuda.is_available()
    if run_on_gpu: model = model.cuda()

    openset_label_embedding = []
    for category in categories:
        texts = [t.format(processed_name(category, rm_dot=True), article=article(category))
                 for t in templates]
        texts = ["This is " + x if x.startswith("a") or x.startswith("the") else x for x in texts]
        tokens = clip.tokenize(texts)
        if run_on_gpu: tokens = tokens.cuda()
        text_embeddings = model.encode_text(tokens)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        text_embedding = F.normalize(text_embeddings.mean(dim=0), dim=0)
        openset_label_embedding.append(text_embedding)
    return torch.stack(openset_label_embedding, dim=0)

@torch.no_grad()
def build_or_load_clip_cache(
    cache_path: str = "./cache/cifar100_clip_ViTL14.pt",
    image_size: int = 224,
    batch_size: int = 256
) -> Dict[str, torch.Tensor]:
    """
    若已存在缓存则读取；否则（且安装了 clip/torchvision）可以现算一份缓存。
    正常使用时，你应当已通过 extract_CLIP_xxx.py 生成好缓存。
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu")
        for k in ["train_feats", "val_feats", "clip_probs_train", "clip_probs_val", "clip_label_embeds"]:
            if k in data and isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(torch.float32)
        for k in ["train_labels", "val_labels", "clip_argmax_train", "clip_argmax_val"]:
            if k in data and isinstance(data[k], torch.Tensor):
                data[k] = data[k].long()
        print(f"[CACHE] Loaded CLIP features from: {cache_path}")
        return data

    if clip is None:
        raise RuntimeError("CLIP package not available. Please prepare cache via extract_CLIP_xxx.py first.")

    print("[CACHE] Building CLIP features...")
    train_data, train_label, test_data, test_label = read_data_cifar_100()
    train_label = torch.tensor(train_label, dtype=torch.long)
    val_label   = torch.tensor(test_label, dtype=torch.long)

    clip_model, _ = clip.load("ViT-L/14")
    clip_model = clip_model.to(device).eval()
    print("[CLIP] Loaded ViT-L/14.")

    info = load_taglist(dataset="CIFAR100")
    label_embeds = build_clip_label_embedding(clip_model, info["taglist"]).to(device).to(torch.float32)

    transform = get_transform(image_size=image_size)
    class CIFARArrayDataset(Dataset):
        def __init__(self, imgs, labels, tfm):
            self.imgs, self.labels, self.tfm = imgs, labels, tfm
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            x = Image.fromarray(np.uint8(self.imgs[idx]).transpose((1, 2, 0)))
            return self.tfm(x), int(self.labels[idx])

    train_loader = DataLoader(CIFARArrayDataset(train_data, train_label, transform),
                              batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(CIFARArrayDataset(test_data,  val_label,   transform),
                              batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def encode_split(loader):
        feats_list, labels_list, probs_list, argmax_list = [], [], [], []
        for xb, yb in loader:
            xb = xb.to(device)
            f = clip_model.encode_image(xb).to(torch.float32)
            f = F.normalize(f, dim=-1)
            logits = f @ label_embeds.t()
            probs  = F.softmax(logits, dim=-1).to(torch.float32)
            yhat   = probs.argmax(dim=-1)

            feats_list.append(f.cpu()); labels_list.append(yb.clone())
            probs_list.append(probs.cpu()); argmax_list.append(yhat.cpu())

        feats  = torch.cat(feats_list, 0).to(torch.float32)
        labels = torch.cat(labels_list, 0).long()
        probs  = torch.cat(probs_list,  0).to(torch.float32)
        argmx  = torch.cat(argmax_list, 0).long()
        return feats, labels, probs, argmx

    train_feats, train_labels, clip_probs_train, clip_argmax_train = encode_split(train_loader)
    val_feats,   val_labels,   clip_probs_val,   clip_argmax_val   = encode_split(val_loader)

    data = {
        "train_feats": train_feats, "train_labels": train_labels,
        "val_feats":   val_feats,   "val_labels":   val_labels,
        "clip_label_embeds": label_embeds.detach().cpu().to(torch.float32),
        "clip_probs_train": clip_probs_train, "clip_probs_val": clip_probs_val,
        "clip_argmax_train": clip_argmax_train, "clip_argmax_val": clip_argmax_val,
    }
    torch.save(data, cache_path)
    print(f"[CACHE] Saved to {cache_path}")
    return data


# -----------------------
# 读取 make_obs_labels.py 的结果
# -----------------------
def load_obs_labels(obs_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Obs labels not found: {obs_path}")
    data = torch.load(obs_path, map_location="cpu")
    if not ("y_obs" in data and "s" in data):
        raise KeyError(f"Obs file must contain 'y_obs' and 's'. Got keys: {list(data.keys())}")
    y_obs = data["y_obs"].long()
    s = data["s"].long()
    print(f"[OBS] Loaded obs labels: y_obs={tuple(y_obs.shape)}, s={tuple(s.shape)}")
    return {"y_obs": y_obs, "s": s}


# -----------------------
# 模型 / 数据集 / 评估
# -----------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x): return self.fc(x)

class FeatureDataset(Dataset):
    def __init__(self, feats, labels):
        self.f = feats; self.y = labels
    def __len__(self): return self.f.size(0)
    def __getitem__(self, idx): return self.f[idx], int(self.y[idx])

@torch.no_grad()
def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048):
    head.eval()
    N = feats.size(0)
    correct, total, ce_sum = 0, 0, 0.0
    for i in range(0, N, batch_size):
        xb = feats[i:i+batch_size].to(device, dtype=torch.float32)
        yb = labels[i:i+batch_size].to(device)
        logits = head(xb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total   += yb.numel()
        ce_sum  += F.cross_entropy(logits, yb, reduction="sum").item()
    return correct / max(1, total), ce_sum / max(1, total)


# -----------------------
# 训练：只用 s=0 子集（强监督，标签用 train_labels 真值）
# -----------------------
def train_strong_supervised(
    cache_path: str = "./cache/cifar100_clip_ViTL14.pt",
    obs_labels_path: str = "./cache/CIFAR100/cifar100_obs_labels.pt",
    batch_size: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 15,
    image_size: int = 224,
    save_path: str = "./outputs/CIFAR100/head_trusted.pt",
    print_every: int = 20,
    label_smoothing: float = 0.0
):
    set_seed(42)

    # 0) 读取或构建 CLIP 特征缓存
    cache = build_or_load_clip_cache(cache_path, image_size=image_size, batch_size=256)
    train_feats = cache["train_feats"]      # (N,D)
    train_labels = cache["train_labels"]    # (N,)
    val_feats   = cache["val_feats"]        # (M,D)
    val_labels  = cache["val_labels"]       # (M,)

    # 单位化（健壮）
    train_feats = F.normalize(train_feats, dim=-1)
    val_feats   = F.normalize(val_feats,   dim=-1)

    C = int(train_labels.max().item()) + 1
    D = int(train_feats.size(1))
    N = int(train_feats.size(0))
    print(f"[INFO] N={N}, D={D}, C={C}")

    # 1) 读取 make_obs_labels.py 的输出，只用 s==0 的可信子集
    obs = load_obs_labels(obs_labels_path)
    y_obs, s_flags = obs["y_obs"], obs["s"]
    if y_obs.numel() != N or s_flags.numel() != N:
        raise RuntimeError(f"Obs length mismatch: feats N={N}, y_obs={y_obs.numel()}, s={s_flags.numel()}")
    trusted_mask = (s_flags == 0)
    num_trusted = int(trusted_mask.sum().item())
    if num_trusted == 0:
        raise RuntimeError("No trusted samples found (s==0). Please check make_obs_labels.py output.")
    print(f"[STEP] Trusted (s=0): {num_trusted} / {N} ({num_trusted / max(1,N):.1%})")

    # 2) 只用可信子集作为训练数据（强监督：使用真值 train_labels）
    feats_trusted  = train_feats[trusted_mask]
    labels_trusted = train_labels[trusted_mask]
    train_loader = DataLoader(FeatureDataset(feats_trusted, labels_trusted),
                              batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # 3) 线性头与优化器
    head = LinearHead(D, C).to(device).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    # 4) 早停（按测试集精度），仅保存最优
    best_state, best_val_acc, wait = None, -1.0, 0

    print("[TRAIN] Strong-supervised (trusted-only) training starts...")
    for epoch in range(1, max_epochs + 1):
        head.train()
        t0 = time.time()
        running_loss, running_batches = 0.0, 0

        for ib, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device)

            logits = head(xb)

            if label_smoothing > 0.0:
                # 可选 label smoothing
                eps = label_smoothing
                target = torch.full_like(logits, eps / (C - 1))
                target.scatter_(1, yb.view(-1,1), 1.0 - eps)
                loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
            else:
                loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_batches += 1

            if ib % print_every == 0:
                print(f"[Epoch {epoch:03d}] batch {ib:04d} | loss(avg)={running_loss/running_batches:.5f}")

        # 5) 评估（全训练集 / 可信子集 / 测试集）
        train_acc_all,  _ = evaluate(head, train_feats, train_labels, batch_size=2048)
        train_acc_trust, _ = evaluate(head, feats_trusted, labels_trusted, batch_size=2048)
        val_acc, val_ce = evaluate(head, val_feats, val_labels, batch_size=2048)
        dt = time.time() - t0

        print(f"[Epoch {epoch:03d}] time={dt:.1f}s | train_loss={running_loss/max(1,running_batches):.5f} | "
              f"TrainAcc(all)={train_acc_all*100:.2f}% | "
              f"TrainAcc(trusted)={train_acc_trust*100:.2f}% | "
              f"TestAcc={val_acc*100:.2f}% | TestCE={val_ce:.4f}")

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(best_state, save_path)
            print(f"[CHECKPOINT] Improved. Saved to {save_path}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("[EARLY] Early stopping triggered.")
                break

    # 6) 载入最优并最终报告
    if best_state is not None:
        head.load_state_dict(best_state)
    train_acc_all,  _ = evaluate(head, train_feats, train_labels, batch_size=2048)
    train_acc_trust, _ = evaluate(head, feats_trusted, labels_trusted, batch_size=2048)
    val_acc, val_ce = evaluate(head, val_feats, val_labels, batch_size=2048)
    print(f"[FINAL] TrainAcc(all)={train_acc_all*100:.2f}% | "
          f"TrainAcc(trusted)={train_acc_trust*100:.2f}% | "
          f"TestAcc={val_acc*100:.2f}% | TestCE={val_ce:.4f}")
    print("[DONE]")


# -----------------------
# main
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train linear head using ONLY trusted (s=0) samples (strong supervision)")
    ap.add_argument("--cache", type=str, default="./cache/CIFAR100/clip_ViTL14.pt")
    ap.add_argument("--obs_labels_path", type=str, default="./cache/CIFAR100/cifar100_obs_labels.pt")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    # 已删除 --top_pct 参数
    ap.add_argument("--save", type=str, default="./outputs/CIFAR100/head_trusted.pt")
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    args = ap.parse_args()

    set_seed(42)
    train_strong_supervised(
        cache_path=args.cache,
        obs_labels_path=args.obs_labels_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.wd,
        max_epochs=args.epochs,
        patience=args.patience,
        save_path=args.save,
        print_every=args.print_every,
        label_smoothing=args.label_smoothing,
    )
