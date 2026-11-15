"""Train paired AUC-C/soft-CE controls with an immutable correction mask.

Each invocation is one independent training seed. The observed-label artifact
must be produced by compare_label_corrections.py, and is never resampled here.
Only features, observed labels, correction flags and VLM probabilities enter
training batches; true training labels are used solely for input integrity.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from contrast.cache_experiment import evaluate
from fcl import auc_loss, train_auc_ce
from fcl.auc_loss import soft_pairwise_auc
from fcl.train_auc_ce import LinearHead, PKBatchSampler, load_clip_cache_strict, load_obs_labels, set_seed
from scripts import compare_label_corrections as correction


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def source_fingerprint() -> str:
    """Hash the executed source dependencies, excluding working data and pycache."""
    paths = [Path(__file__), Path(train_auc_ce.__file__), Path(auc_loss.__file__),
             ROOT / "src/contrast/cache_experiment.py", Path(correction.__file__),
             ROOT / "scripts/run_paper_experiments.py", ROOT / "src/paths.py"]
    return correction.json_sha256({str(path.relative_to(ROOT)).replace("\\", "/"):
                                   correction.file_sha256(path) for path in paths})


def state_fingerprint(state: dict) -> str:
    return correction.json_sha256({name: correction.tensor_sha256(value)
                                   for name, value in sorted(state.items())})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--obs-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=("fcl_auc", "ce_soft"), required=True)
    parser.add_argument("--dataset", help="Optional assertion against the observed-label metadata")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--P", type=int, default=32)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--lr-max", type=float, default=.005)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--lam", type=float, default=1.)
    parser.add_argument("--mu", type=float, default=.5)
    parser.add_argument("--tau", type=float, default=.5)
    parser.add_argument("--patience", type=int, default=0, choices=(0,),
                        help="Fixed-epoch comparisons disable early stopping")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0 or args.epochs <= 0 or args.P <= 0 or args.K <= 0:
        raise ValueError("Seed must be nonnegative and epochs/P/K positive")
    if args.num_workers < 0 or args.eval_batch_size <= 0 or args.patience != 0:
        raise ValueError("Invalid workers/evaluation batch size; fixed-epoch runs require patience=0")
    for name in ("lr", "lr_max", "tau"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "lam"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.mu) or not 0 <= args.mu <= 1:
        raise ValueError("mu must be finite and in [0,1]")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty run directory: {args.output_dir}")


def load_fixed_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """Verify the mask artifact against the exact cache and sample ordering."""
    cache_sha = correction.file_sha256(args.cache)
    obs_sha = correction.file_sha256(args.obs_labels)
    cache = load_clip_cache_strict(str(args.cache))
    count, classes = cache["clip_probs_train"].shape
    observed = torch.load(args.obs_labels, map_location="cpu", weights_only=True)
    checked = load_obs_labels(str(args.obs_labels), expected_size=count, num_classes=classes)
    if not isinstance(observed.get("meta"), dict):
        raise ValueError("Fixed observed labels require provenance in 'meta'")
    meta = observed["meta"]
    for key in ("dataset", "label_source", "method", "cache_sha256", "mask_sha256", "sample_order_sha256"):
        if not meta.get(key):
            raise ValueError(f"Observed-label metadata lacks {key}")
    if args.dataset and args.dataset != meta["dataset"]:
        raise ValueError("Requested dataset differs from the fixed observed-label metadata")
    if meta["cache_sha256"] != cache_sha:
        raise ValueError("Observed-label cache SHA does not match --cache")
    if meta["label_source"] not in ("clip", "qwen"):
        raise ValueError("Observed-label source must be clip or qwen")
    predictions, _, _ = correction.validate_cache(cache, meta["label_source"], meta.get("prompt") or "C")
    pseudo = observed.get("y_pseudo")
    if not isinstance(pseudo, torch.Tensor) or not torch.equal(pseudo, predictions["train"]):
        raise ValueError("Observed artifact original pseudo labels differ from cache predictions")
    selected = observed.get("selected_mask")
    if (not isinstance(selected, torch.Tensor) or selected.dtype != torch.bool
            or selected.shape != (count,) or not torch.equal(selected, checked["s"].eq(0))):
        raise ValueError("Selected mask must be boolean and agree with s=0")
    if not torch.equal(checked["y_obs"][~selected], pseudo[~selected]):
        raise ValueError("Uncorrected observed labels differ from the original pseudo labels")
    if not torch.equal(checked["y_obs"][selected], cache["train_labels"][selected]):
        raise ValueError("Corrected labels must match selected oracle ground truth")
    sample_ids, _ = correction.stable_sample_order(cache, count, cache_sha)
    if observed.get("train_sample_ids") != sample_ids:
        raise ValueError("Observed-label sample identities/order differ from cache")
    row_indices = observed.get("train_row_indices")
    if not isinstance(row_indices, torch.Tensor) or not torch.equal(row_indices, torch.arange(count)):
        raise ValueError("Observed-label row indices are not the original cache order")
    hashes = {"cache_sha256": cache_sha, "obs_sha256": obs_sha,
              "mask_sha256": correction.tensor_sha256(selected),
              "y_obs_sha256": correction.tensor_sha256(checked["y_obs"]),
              "pseudo_sha256": correction.tensor_sha256(pseudo),
              "sample_order_sha256": correction.json_sha256(sample_ids)}
    for key in ("mask_sha256", "sample_order_sha256", "y_obs_sha256"):
        if key in meta and hashes[key] != meta[key]:
            raise ValueError(f"Observed-label {key} does not match its data")
    return cache, {**checked, "meta": meta, "selected_mask": selected}, hashes


def batch_objective(logits, y_obs, s, probabilities, *, objective, lam, mu, tau):
    """Reduce each subset separately, preserving the recovered FCL weighting."""
    trusted, noisy = s.eq(0), s.eq(1)
    trusted_loss = F.cross_entropy(logits[trusted], y_obs[trusted]) if trusted.any() else logits.sum() * 0
    if objective == "fcl_auc":
        untrusted_loss = soft_pairwise_auc(logits, probabilities, s, mu=mu, tau=tau)
    elif objective == "ce_soft":
        untrusted_loss = (-(probabilities[noisy].detach() * logits[noisy].log_softmax(-1)).sum(-1).mean()
                          if noisy.any() else logits.sum() * 0)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return trusted_loss + lam * untrusted_loss, trusted_loss, untrusted_loss


def common_protocol(args: argparse.Namespace) -> dict:
    """Shared comparison protocol deliberately excludes seed/objective/paths."""
    return {"version": 1, "epochs": args.epochs, "patience": 0, "P": args.P, "K": args.K,
            "lr": args.lr, "lr_max": args.lr_max, "weight_decay": args.weight_decay,
            "lam": args.lam, "mu": args.mu, "tau": args.tau, "num_workers": args.num_workers,
            "eval_batch_size": args.eval_batch_size, "head": "xavier_uniform_linear_with_zero_bias",
            "feature_normalization": "L2_per_row", "optimizer": "AdamW",
            "scheduler": {"name": "OneCycleLR", "pct_start": .15, "anneal_strategy": "cos",
                          "div_factor": 10., "final_div_factor": 10.},
            "gradient_clip_norm": 5., "sampler": "PK_grouped_by_observed_labels",
            "loss_reduction": "mean_each_subset_then_CE_plus_lam_times_untrusted",
            "checkpoint_selection": "highest_validation_accuracy_first_tie",
            "test_policy": "one_evaluation_after_validation_checkpoint_selection",
            "deterministic_algorithms": True, "tf32": False}


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    cache, observed, hashes = load_fixed_inputs(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features = {split: F.normalize(cache[f"{split}_feats"].float(), dim=-1)
                for split in ("train", "val", "test")}
    # AUC-C uses the selected VLM prior; the soft-CE control always uses CLIP.
    prior_key = ("original_clip_probs_train" if args.objective == "ce_soft"
                 and "original_clip_probs_train" in cache else "clip_probs_train")
    if args.objective == "ce_soft" and observed["meta"]["label_source"] != "clip" and prior_key == "clip_probs_train":
        raise ValueError("Soft-CE control requires original CLIP probabilities")
    prior = train_auc_ce._probability_tensor(cache[prior_key], prior_key, len(features["train"]),
                                           cache["clip_probs_train"].shape[1])
    # Row indices are included only to fingerprint the actual sampling stream.
    train_data = TensorDataset(features["train"], observed["y_obs"], observed["s"], prior,
                               torch.arange(len(prior)))
    sampler = PKBatchSampler(observed["y_obs"], args.P, args.K, seed=args.seed)
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
    model = LinearHead(features["train"].shape[1], prior.shape[1]).to(device).float()
    hashes["init_state_sha256"] = state_fingerprint(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_max, total_steps=len(loader) * args.epochs,
        pct_start=.15, anneal_strategy="cos", div_factor=10., final_div_factor=10.)
    protocol = common_protocol(args)
    hashes.update(source_sha256=source_fingerprint(), protocol_sha256=correction.json_sha256(protocol),
                  prior_sha256=correction.tensor_sha256(prior))
    meta = observed["meta"]
    identity = {"schema_version": 1, "dataset": meta["dataset"], "label_source": meta["label_source"],
                "selection": meta.get("selection", meta["method"]), "selection_seed": meta.get("selection_seed"),
                "objective": args.objective, "seed": args.seed}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership also protects against two jobs racing for an empty directory.
    with (args.output_dir / ".run.lock").open("x", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    args._owns_output = True
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    config = {**identity, "protocol": protocol, "fingerprints": dict(hashes),
              "cache": str(args.cache.resolve()), "observed_labels": str(args.obs_labels.resolve()),
              "output_dir": str(args.output_dir.resolve()), "prior_field": prior_key,
              "fixed_mask_metadata": meta, "corrected_count": int(observed["selected_mask"].sum()),
              "training_count": len(prior), "steps_per_epoch": len(loader),
              "git_commit": commit, "started_at": started,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda_runtime": torch.version.cuda, "device": str(device),
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}
    write_json(args.output_dir / "config.json", config)
    hashes["config_sha256"] = correction.file_sha256(args.output_dir / "config.json")
    write_json(args.output_dir / "status.json", {**identity, "status": "running", "completed_epochs": 0})
    stream_hash = hashlib.sha256()
    best_accuracy, best_epoch = -math.inf, 0
    best_metrics, history = None, []
    began = time.monotonic()
    with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "trusted_ce", "untrusted_loss",
                                                   "val_accuracy", "val_nll", "lr", "seconds", "sampler_sha256"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            sampler.set_epoch(epoch - 1)
            model.train()
            totals = [0., 0., 0.]
            epoch_start = time.monotonic()
            stream_hash.update(f"epoch:{epoch};".encode("ascii"))
            for batch_index, (xb, yb, flags, pb, indices) in enumerate(loader, start=1):
                stream_hash.update(indices.numpy().tobytes())
                logits = model(xb.to(device))
                losses = batch_objective(logits, yb.to(device), flags.to(device), pb.to(device),
                                         objective=args.objective, lam=args.lam, mu=args.mu, tau=args.tau)
                if not all(bool(torch.isfinite(value)) for value in losses):
                    raise FloatingPointError(f"Nonfinite objective at epoch={epoch}, batch={batch_index}")
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                optimizer.step()
                scheduler.step()
                totals = [total + float(value.detach()) for total, value in zip(totals, losses)]
            validation = evaluate(model, features["val"], cache["val_labels"], device,
                                  batch_size=args.eval_batch_size)
            if not math.isfinite(validation["accuracy"]) or not math.isfinite(validation["nll"]):
                raise FloatingPointError(f"Nonfinite validation metrics at epoch={epoch}")
            row = {"epoch": epoch, "train_loss": totals[0] / len(loader),
                   "trusted_ce": totals[1] / len(loader), "untrusted_loss": totals[2] / len(loader),
                   "val_accuracy": validation["accuracy"], "val_nll": validation["nll"],
                   "lr": scheduler.get_last_lr()[0], "seconds": time.monotonic() - epoch_start,
                   "sampler_sha256": stream_hash.hexdigest()}
            history.append(row)
            writer.writerow(row)
            handle.flush()
            if validation["accuracy"] > best_accuracy:
                best_accuracy, best_epoch, best_metrics = validation["accuracy"], epoch, validation
                torch.save(model.state_dict(), args.output_dir / "best.ckpt")
            torch.save(model.state_dict(), args.output_dir / "last.ckpt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()}, args.output_dir / "last_training_state.pt")
            write_json(args.output_dir / "status.json", {**identity, "status": "running", "completed_epochs": epoch,
                       "best_epoch": best_epoch, "best_val_accuracy": best_accuracy})
            print(f"[EPOCH] {meta['dataset']}/{identity['selection']}/{args.objective}/seed{args.seed} "
                  f"{epoch}/{args.epochs} loss={row['train_loss']:.6f} val_acc={100*validation['accuracy']:.4f}%", flush=True)
    model.load_state_dict(torch.load(args.output_dir / "best.ckpt", map_location=device, weights_only=True))
    test_metrics = evaluate(model, features["test"], cache["test_labels"], device,
                            batch_size=args.eval_batch_size)
    if test_metrics["valid_count"] != len(cache["test_labels"]):
        raise ValueError("Final test metrics are incomplete")
    if not all(math.isfinite(test_metrics[key]) for key in ("accuracy", "nll", "macro_f1", "ece")):
        raise FloatingPointError("Nonfinite final test metrics")
    hashes["sampler_sha256"] = stream_hash.hexdigest()
    hashes["best_ckpt_sha256"] = correction.file_sha256(args.output_dir / "best.ckpt")
    hashes["last_ckpt_sha256"] = correction.file_sha256(args.output_dir / "last.ckpt")
    result = {**identity, "status": "completed", "completed_epochs": len(history),
              "best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
              "best_val": best_metrics, "test_accuracy": test_metrics["accuracy"], "test": test_metrics,
              "fingerprints": hashes, "protocol": protocol, "prior_field": prior_key,
              "corrected_count": config["corrected_count"], "training_count": len(prior),
              "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "elapsed_seconds": time.monotonic() - began, "git_commit": commit}
    write_json(args.output_dir / "metrics.json", result)
    write_json(args.output_dir / "status.json", {**identity, "status": "completed", "completed_epochs": len(history)})
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        # Never write into a pre-existing directory before input validation.
        if getattr(args, "_owns_output", False) and not (args.output_dir / "metrics.json").exists():
            write_json(args.output_dir / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                       "failed_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        raise
    print(json.dumps({"output_dir": str(args.output_dir), "test_accuracy": result["test_accuracy"],
                      "best_epoch": result["best_epoch"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
