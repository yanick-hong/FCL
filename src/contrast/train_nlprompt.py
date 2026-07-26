"""Cache-compatible NLPrompt adaptation.

The original CVPR 2025 method learns prompts with CLIP.  This implementation
keeps the cached CLIP image features fixed and learns the class text prototypes
stored in the cache.  Trusted samples use CE and the remaining observed labels
use PromptMAE's MAE objective.  No image is re-encoded during training.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from contrast.cache_experiment import (evaluate, load_cache_and_observed,
                                       make_run_dir, save_config, save_history,
                                       save_result, save_run_metadata)
from paths import clip_cache, observed_labels_cache


class CachedPromptHead(nn.Module):
    def __init__(self, init_prototypes: torch.Tensor, num_classes: int):
        super().__init__()
        if init_prototypes.ndim == 2 and init_prototypes.shape[0] == num_classes:
            prototypes = init_prototypes.float()
        else:
            prototypes = torch.randn(num_classes, init_prototypes.shape[-1])
        self.prototypes = nn.Parameter(F.normalize(prototypes, dim=-1))
        self.logit_scale = nn.Parameter(torch.tensor(4.0))  # exp ~= 54.6
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image = F.normalize(features.float(), dim=-1)
        text = F.normalize(self.prototypes.float(), dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * image @ text.t() + self.bias


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--obs-labels", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache_path = args.cache or str(clip_cache(args.dataset))
    obs_path = args.obs_labels or str(observed_labels_cache(args.dataset))
    run_dir = make_run_dir(args.dataset, "nlprompt", args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_cache_and_observed(cache_path, obs_path)
    train_x, val_x, test_x = bundle["train_feats"], bundle["val_feats"], bundle["test_feats"]
    train_y, val_y, test_y = bundle["train_labels"], bundle["val_labels"], bundle["test_labels"]
    y_obs, trusted = bundle["y_obs"], bundle["s"].eq(0)
    classes = int(bundle["num_classes"])
    prototypes = bundle.get("clip_label_embeds", torch.randn(classes, train_x.shape[1]))
    model = CachedPromptHead(prototypes, classes).to(device)
    loader = DataLoader(TensorDataset(train_x, train_y, y_obs, trusted),
                        batch_size=args.batch_size, shuffle=True, num_workers=2,
                        pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_acc, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for features, true_y, observed_y, is_trusted in loader:
            features = features.to(device, non_blocking=True)
            true_y = true_y.to(device)
            observed_y = observed_y.to(device)
            is_trusted = is_trusted.to(device)
            logits = model(features)
            clean_loss = F.cross_entropy(logits[is_trusted], true_y[is_trusted]) if is_trusted.any() else logits.sum() * 0
            noisy_logits = logits[~is_trusted]
            if noisy_logits.shape[0]:
                target = F.one_hot(observed_y[~is_trusted], classes).float()
                mae_loss = (noisy_logits.softmax(dim=-1) - target).abs().mean()
            else:
                mae_loss = logits.sum() * 0
            loss = clean_loss + args.mae_weight * mae_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())
        scheduler.step()
        val_metrics = evaluate(model, val_x, val_y, device)
        row = {"epoch": epoch, "train_loss": running / max(1, len(loader)), **val_metrics}
        history.append(row)
        print(f"[NLPrompt][{args.dataset}] epoch={epoch:03d} val_acc={val_metrics['accuracy']*100:.2f}%")
        if val_metrics["accuracy"] > best_acc:
            best_acc, best_epoch, bad_epochs = val_metrics["accuracy"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, run_dir / "best.ckpt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, run_dir / "last.ckpt")
    save_history(run_dir, history)
    test_metrics = evaluate(model, test_x, test_y, device)
    save_config(run_dir, {"method": "NLPrompt-cache-adapted", "dataset": args.dataset,
                          "cache": cache_path, "observed_labels": obs_path,
                          "args": vars(args), "trusted_ratio": float(trusted.float().mean())})
    save_result(run_dir, {"method": "NLPrompt-cache-adapted", "dataset": args.dataset,
                          "best_val": history[max(0, best_epoch - 1)] if history else {},
                          "best_val_accuracy": best_acc, "best_epoch": best_epoch,
                          "test": test_metrics, "trusted_ratio": float(trusted.float().mean()),
                          "elapsed_seconds": time.time() - start_time})
    save_run_metadata(run_dir, method="NLPrompt-cache-adapted", dataset=args.dataset,
                      script="src/contrast/train_nlprompt.py", args=args,
                      command=sys.argv)
    print(f"[NLPrompt][DONE] {run_dir}")


if __name__ == "__main__":
    main()
