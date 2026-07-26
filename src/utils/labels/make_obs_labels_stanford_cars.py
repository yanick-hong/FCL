"""Generate observed labels for the Stanford Cars CLIP cache."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from paths import clip_cache, observed_labels_cache
from utils.labels.make_obs_labels_cifar100 import make_observed_labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_path", default=str(clip_cache("stanford_cars")))
    p.add_argument("--save_path", default=str(observed_labels_cache("stanford_cars")))
    p.add_argument("--top_outlier_pct", type=float, default=0.30)
    a = p.parse_args(); c = torch.load(a.cache_path, map_location="cpu")
    pseudo = c.get("clip_argmax_train", c["clip_probs_train"].argmax(dim=1))
    y_obs, s = make_observed_labels(c["train_feats"], c["train_labels"], pseudo, a.top_outlier_pct)
    out = Path(a.save_path); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"y_obs": y_obs.long(), "s": s.long(), "meta": {"dataset": "Stanford Cars", "from_cache": str(Path(a.cache_path).resolve()), "top_outlier_pct": a.top_outlier_pct}}, out)
    print(f"[DONE] Saved observed labels to {out}; trusted={(s == 0).sum().item()}")

if __name__ == "__main__":
    main()
