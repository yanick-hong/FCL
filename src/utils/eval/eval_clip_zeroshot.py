# -*- coding: utf-8 -*-
'''
eval_clip_zeroshot_strict.py

强制零样本评估（strict zero-shot）：
- 优先用 feats_* 与 clip_label_embeds（或 clip_label_embedding）计算余弦相似度 => top-k
- 不使用/不信任任何 head_* / trained_* / logits_head_* 等训练产物
- 若无文本标签嵌入但有 clip_probs_*（零样本概率），则退化用它计算准确率

用法：
  python src/utils/eval/eval_clip_zeroshot.py "${DATA_ROOT}/processed/*_clip_*.pt" --glob --out_csv "${OUTPUT_ROOT}/zeroshot_strict.csv"
  python src/utils/eval/eval_clip_zeroshot.py "${DATA_ROOT}/processed/cifar100_clip_vit-l-14_embeddings.pt"
'''
import os, sys, glob, argparse
from typing import Dict, List, Tuple, Optional
import torch
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import OUTPUT_ROOT

def parse_args():
    p = argparse.ArgumentParser("Strict zero-shot CLIP accuracy evaluator")
    p.add_argument("paths", nargs="+", help="缓存路径；配合 --glob 可使用通配符")
    p.add_argument("--glob", action="store_true", help="将 paths 作为通配模式展开（加引号避免 shell 展开）")
    p.add_argument("--splits", type=str, default="train,val,test", help="要评估的 split，逗号分隔")
    p.add_argument("--topk", type=str, default="1,5", help="top-k 列表，逗号分隔")
    p.add_argument("--chunksz", type=int, default=4096, help="分块计算相似度的 batch 大小")
    p.add_argument("--out_csv", type=str, default=str(OUTPUT_ROOT / "zeroshot_strict.csv"), help="结果 CSV 输出路径")
    p.add_argument("--sanity", action="store_true", help="若存在 clip_argmax_*，与本脚本结果做一致性检查（只做对比，不影响结果）")
    return p.parse_args()

def _nice_name(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}/{stem}" if parent and parent.lower() != "cache" else stem

def _get_label_embeds(data: Dict) -> Optional[torch.Tensor]:
    for k in ["clip_label_embeds", "clip_label_embedding"]:
        if k in data:
            emb = data[k]
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb)
            return emb.to(torch.float32)
    return None

def _get_feats_and_labels(data: Dict, sp: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    feats, labels = None, None
    fk = f"{sp}_feats"
    lk = f"{sp}_labels"
    if fk in data and lk in data:
        feats = data[fk]
        labels = data[lk]
        if not isinstance(feats, torch.Tensor): feats = torch.as_tensor(feats)
        if not isinstance(labels, torch.Tensor): labels = torch.as_tensor(labels)
        feats = feats.to(torch.float32)
        labels = labels.to(torch.long).view(-1)
        return feats, labels
    return None, None

def _get_clip_probs(data: Dict, sp: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    pk = f"clip_probs_{sp}"
    lk = f"{sp}_labels"
    if pk in data and lk in data:
        probs = data[pk]
        labels = data[lk]
        if not isinstance(probs, torch.Tensor): probs = torch.as_tensor(probs)
        if not isinstance(labels, torch.Tensor): labels = torch.as_tensor(labels)
        return probs.to(torch.float32), labels.to(torch.long).view(-1)
    return None

@torch.no_grad()
def _topk_acc_from_probs(probs: torch.Tensor, labels: torch.Tensor, ks: List[int]) -> Dict[int, float]:
    N, C = probs.shape
    ks = [min(max(1, k), C) for k in ks]
    maxk = max(ks)
    _, pred = probs.topk(maxk, dim=1, largest=True, sorted=True)  # (N,maxk)
    labels_exp = labels.view(-1,1).expand(-1,maxk)
    correct = pred.eq(labels_exp)
    res = {}
    for k in ks:
        res[k] = correct[:, :k].any(dim=1).float().mean().item()
    return res

@torch.no_grad()
def _topk_acc_by_cosine(feats: torch.Tensor, label_emb: torch.Tensor, labels: torch.Tensor,
                        ks: List[int], chunksz: int = 4096, sanity_with_argmax: Optional[torch.Tensor] = None):
    # 统一归一化（CLIP 通常在 img/text 特征上用 L2 norm；温度缩放不影响排序）
    feats = torch.nn.functional.normalize(feats, dim=1)
    label_emb = torch.nn.functional.normalize(label_emb, dim=1)
    N, C = feats.size(0), label_emb.size(0)
    ks = [min(max(1,k), C) for k in ks]
    maxk = max(ks)

    correct_k = {k: 0 for k in ks}
    mismatch_cnt = 0
    total = 0

    for i in range(0, N, chunksz):
        x = feats[i:i+chunksz]       # (b,D)
        sims = x @ label_emb.t()     # (b,C) 余弦相似度
        _, pred = sims.topk(maxk, dim=1, largest=True, sorted=True)  # (b,maxk)

        # sanity: 与缓存里的 argmax 比一比（若有）
        if sanity_with_argmax is not None:
            argm = pred[:, 0]
            gt_argm = sanity_with_argmax[i:i+chunksz].view(-1)
            mismatch_cnt += (argm != gt_argm).sum().item()

        y = labels[i:i+chunksz].view(-1,1)  # (b,1)
        for k in ks:
            hit = (pred[:, :k] == y).any(dim=1).sum().item()
            correct_k[k] += hit
        total += y.size(0)

    accs = {k: correct_k[k] / max(1,total) for k in ks}
    sanity_rate = (mismatch_cnt / max(1,total)) if sanity_with_argmax is not None else None
    return accs, total, sanity_rate

def main():
    args = parse_args()
    if args.glob:
        paths = []
        for pat in args.paths: paths.extend(glob.glob(pat, recursive=True))
        paths = sorted(set([p for p in paths if os.path.isfile(p)]))
    else:
        paths = [p for p in args.paths if os.path.isfile(p)]
    if not paths:
        print("[ERR] 未找到缓存文件"); sys.exit(1)

    want_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    ks = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["dataset","split","N"] + [f"top{k}" for k in ks]
        w.writerow(hdr)

        for path in paths:
            try:
                data = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"[WARN] 载入失败: {path} | {e}"); continue

            ds_name = _nice_name(path)

            # 强制忽略任何训练头提示
            suspicious_keys = [k for k in data.keys() if ("head" in k.lower() or "linear" in k.lower())]
            if suspicious_keys:
                print(f"[INFO] 忽略训练相关键: {ds_name} -> {suspicious_keys}")

            label_emb = _get_label_embeds(data)

            for sp in want_splits:
                # 先尝试严格零样本路径：feats + label_embeds
                feats, labels = _get_feats_and_labels(data, sp)
                row_written = False

                if feats is not None and labels is not None and label_emb is not None:
                    sanity_argmax = data.get(f"clip_argmax_{sp}") if args.sanity else None
                    if isinstance(sanity_argmax, torch.Tensor):
                        sanity_argmax = sanity_argmax.to(torch.long).view(-1)

                    accs, N, sanity_rate = _topk_acc_by_cosine(
                        feats, label_emb, labels, ks, chunksz=args.chunksz,
                        sanity_with_argmax=sanity_argmax
                    )
                    print(f"[ZS] {ds_name:<30} | {sp:<5} | N={N:<6} | " +
                          " ".join([f"top{k}:{accs[k]*100:.2f}%" for k in ks]) +
                          (f" | sanity mismatch:{sanity_rate*100:.2f}%" if sanity_rate is not None else ""))
                    w.writerow([ds_name, sp, N] + [f"{accs[k]:.6f}" for k in ks])
                    row_written = True

                # 退化路径：没有 label_embeds，但有 clip_probs_*（依旧是零样本）
                if not row_written:
                    pp = _get_clip_probs(data, sp)
                    if pp is not None:
                        probs, labels2 = pp
                        accs = _topk_acc_from_probs(probs, labels2, ks)
                        N = labels2.numel()
                        print(f"[ZS*] {ds_name:<30} | {sp:<5} | N={N:<6} | " +
                              " ".join([f"top{k}:{accs[k]*100:.2f}%" for k in ks]) +
                              " | via clip_probs")
                        w.writerow([ds_name, sp, N] + [f"{accs[k]:.6f}" for k in ks])
                        row_written = True

                if not row_written:
                    print(f"[WARN] {ds_name}:{sp} 缺少零样本所需键（feats+label_embeds 或 clip_probs_*），已跳过。")

    print(f"\n[Done] 严格零样本评估完成，结果保存在: {args.out_csv}")

if __name__ == "__main__":
    main()
