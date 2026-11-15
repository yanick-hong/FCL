"""Plan, audit and run the experiments explicitly reported in the FCL paper.

Dry-run is the default. Heavy libraries are imported only for preflight/execution.
FL-C/NL-C/FS-C below are exact CE controls; other learners use existing CLIs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from paths import OUTPUT_ROOT, PROCESSED_ROOT

METHODS = ("zero_shot", "fl_c", "nl_c", "fs_c", "l2b", "dld", "csgn", "fcl")
SUITES = ("main", "lambda", "rho", "prompts", "epochs")

_SOURCE_FINGERPRINT: tuple[str, bool] | None = None


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/paper_reproduction.yaml")
    p.add_argument("--suite", choices=(*SUITES, "all"), default="main")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    p.add_argument("--label-source", choices=("clip", "qwen"), default="clip")
    p.add_argument("--prompt", choices=("A", "B", "C"), default="C")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--cache-template", default=None,
                   help="Override source cache template; fields: processed,dataset,source,prompt,rho.")
    p.add_argument("--obs-template", default=None)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "paper")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers for FCL and internal CE runners (default: 0).")
    p.add_argument("--P", type=int, default=None, help="Override classes per sampled batch.")
    p.add_argument("--K", type=int, default=None, help="Override samples per class.")
    p.add_argument("--pairwise-tau", type=float, default=None,
                   help="Override FCL pairwise logistic temperature.")
    p.add_argument("--resume", action="store_true",
                   help="Skip completed jobs whose input/source fingerprints still match.")
    p.add_argument("--prepare-labels", action="store_true",
                   help="Create missing observed labels from the selected source cache (oracle correction simulation).")
    p.add_argument("--preflight", action="store_true", help="Load and audit all selected inputs, without training.")
    p.add_argument("--strict-paper-splits", action="store_true",
                   help="Reject a training count/class count different from Table 1.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def build_jobs(cfg: dict, args: argparse.Namespace) -> list[dict]:
    if args.datasets:
        unknown = set(args.datasets) - set(cfg["datasets"])
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    jobs = []
    protocol = cfg["protocol"]
    for suite in (SUITES if args.suite == "all" else (args.suite,)):
        ablation = cfg.get("ablations", {}).get(suite, {})
        datasets = args.datasets or ablation.get("datasets", list(cfg["datasets"]))
        if suite != "main":
            datasets = [d for d in datasets if d in ablation.get("datasets", datasets)]
        methods = args.methods or (list(METHODS) if suite in ("main", "epochs") else ["fcl"])
        values = ablation.get("values", [None])
        for dataset in datasets:
            for value in values:
                source = "qwen" if suite == "prompts" else args.label_source
                prompt = str(value) if suite == "prompts" else args.prompt
                rho = float(value) if suite == "rho" else float(protocol["rho"])
                lam = float(value) if suite == "lambda" else float(protocol["lam"])
                classes_per_batch = args.P if args.P is not None else int(cfg["datasets"][dataset]["P"])
                samples_per_class = args.K if args.K is not None else int(cfg["datasets"][dataset]["K"])
                paper_classes = int(cfg["datasets"][dataset]["paper_classes"])
                if classes_per_batch <= 0 or classes_per_batch > paper_classes:
                    raise ValueError(
                        f"P={classes_per_batch} is invalid for {dataset}: "
                        f"the sampler cannot draw more than {paper_classes} classes"
                    )
                if samples_per_class <= 0:
                    raise ValueError(f"K={samples_per_class} must be positive for {dataset}")
                pairwise_tau = (args.pairwise_tau if args.pairwise_tau is not None
                                else float(protocol.get("pairwise_tau", 0.5)))
                for method in methods:
                    # The oracle result is independent of source labels.
                    for seed in args.seeds or protocol["seeds"]:
                        fields = dict(processed=str(PROCESSED_ROOT), dataset=dataset,
                                      source=source, prompt=prompt, rho=f"{rho:g}")
                        source_cfg = cfg["sources"][source]
                        cache = Path((args.cache_template or source_cfg["cache_template"]).format(**fields))
                        obs = Path((args.obs_template or source_cfg["obs_template"]).format(**fields))
                        setting = f"prompt-{prompt}_rho-{rho:g}_lam-{lam:g}"
                        run_dir = (args.output_root / suite / source / dataset / method /
                                   setting / f"seed-{seed}" /
                                   dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
                        jobs.append(dict(suite=suite, dataset=dataset, method=method,
                                         source=source, prompt=prompt, rho=rho, lam=lam,
                                         seed=int(seed), cache=cache, obs=obs, run_dir=run_dir,
                                         P=classes_per_batch, K=samples_per_class,
                                         pairwise_tau=pairwise_tau, num_workers=args.num_workers,
                                         protocol_settings=dict(protocol),
                                         epochs=args.epochs or int(ablation.get("epochs", protocol["epochs"]))))
    return jobs


def command_for(job: dict, cfg: dict) -> list[str] | None:
    method = job["method"]
    if method in ("zero_shot", "fl_c", "nl_c", "fs_c"):
        return None
    settings = cfg["protocol"]
    script = {"fcl": "src/fcl/train_auc_ce.py", "l2b": "src/contrast/train_l2b.py",
              "dld": "src/contrast/train_dld.py", "csgn": "src/contrast/train_csgn.py"}[method]
    epoch_arg = "--epochs" if method == "csgn" else "--max_epochs"
    cache_arg = "--cache_path" if method == "csgn" else "--cache"
    command = [sys.executable, str(ROOT / script), cache_arg, str(job["cache"]),
               "--obs_labels_path", str(job["obs"]), epoch_arg, str(job["epochs"]),
               "--P", str(job["P"]), "--K", str(job["K"]),
               "--lr", str(settings["lr"]), "--weight_decay", str(settings["weight_decay"]),
               "--patience", str(job["epochs"] + 1), "--seed", str(job["seed"]),
               "--best_ckpt", str(job["run_dir"] / "best.ckpt"),
               "--val_dir", str(job["run_dir"] / "logs")]
    if method in ("l2b", "dld", "csgn"):
        command += ["--num_workers", str(job["num_workers"])]
    if method in ("fcl", "dld"):
        command += ["--lr_max", str(settings["lr_max"])]
    if method == "fcl":
        command += ["--lam", str(job["lam"]), "--mu", str(settings["mu"]),
                    "--tau", str(job["pairwise_tau"]), "--num_workers", str(job["num_workers"]),
                    "--experiment_name", str(job["run_dir"])]
    return command


def sha256_file(path: Path) -> str | None:
    """Return a stable file digest, or ``None`` when the file is absent."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint() -> tuple[str, bool]:
    """Hash tracked source/config files and report whether the checkout is dirty."""
    global _SOURCE_FINGERPRINT
    if _SOURCE_FINGERPRINT is not None:
        return _SOURCE_FINGERPRINT
    try:
        listed = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.splitlines()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        listed = [str(path.relative_to(ROOT)) for base in (ROOT / "src", ROOT / "scripts", ROOT / "configs")
                  if base.exists() for path in base.rglob("*") if path.is_file()]
        dirty = True
    # Include untracked recovery files too: during reconstruction the runner
    # itself may not be committed yet, but changing it must invalidate resume.
    listed_paths = {ROOT / rel for rel in listed}
    for base in (ROOT / "src", ROOT / "scripts", ROOT / "configs"):
        if base.exists():
            listed_paths.update(path for path in base.rglob("*") if path.is_file())
    paths = [path for path in listed_paths if path.suffix.lower() in
             {".py", ".yaml", ".yml", ".json", ".toml"}]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path) or ""))
    _SOURCE_FINGERPRINT = digest.hexdigest(), dirty
    return _SOURCE_FINGERPRINT


def job_identity(job: dict) -> dict:
    """Fields that identify a job, excluding its timestamped output directory."""
    return {key: (str(value) if isinstance(value, Path) else value)
            for key, value in job.items() if key != "run_dir"}


def job_fingerprint(job: dict, source_sha: str, git_dirty: bool | None = None) -> tuple[str, dict]:
    fingerprints = {"cache_sha256": sha256_file(job["cache"]),
                    "obs_sha256": sha256_file(job["obs"]),
                    "source_sha256": source_sha}
    payload = {"job": job_identity(job), "fingerprints": fingerprints,
               "git_dirty": git_dirty}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), fingerprints


def find_completed_job(output_root: Path, identity: dict, fingerprint: str) -> Path | None:
    """Find a prior completed run with the exact same job and fingerprints."""
    from scripts.summarize_paper_results import metric_accuracy
    if not output_root.exists():
        return None
    for manifest_path in output_root.rglob("paper_run.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("status") == "completed" and payload.get("job_key") == identity \
                and payload.get("fingerprint") == fingerprint:
            metrics_path = manifest_path.parent / "metrics.json"
            if not metrics_path.is_file():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(metrics, dict) or metric_accuracy(metrics) is None:
                continue
            return manifest_path.parent
    return None


def torch_load(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def qwen_split_metadata(cache: dict, split: str) -> dict | None:
    """Prove the compatibility clip_* fields were replaced for this split.

    A global source string cannot establish that an exporter overlaid test as
    well as train; the remaining clip_* fields may still be genuine CLIP.
    """
    all_metadata = cache.get("annotation_metadata", {})
    if not isinstance(all_metadata, dict):
        return None
    metadata = all_metadata.get(split, {})
    if not isinstance(metadata, dict):
        return None
    models = metadata.get("source_models", [])
    if (metadata.get("replaced_clip_fields") is True and isinstance(models, (list, tuple)) and models
            and all(str(model).lower().rsplit("/", 1)[-1].startswith("qwen") for model in models)):
        return metadata
    return None


def prediction_splits(job: dict, cache: dict) -> tuple[str, ...]:
    if job["method"] != "zero_shot":
        return ("train",)
    if job["source"] == "qwen":
        # Report only Qwen-labelled splits. In particular never label leftover
        # CLIP validation/test predictions as Qwen zero-shot measurements.
        return tuple(split for split in ("train", "val", "test") if qwen_split_metadata(cache, split))
    return ("train", "val", "test")


def inspect_input(job: dict, cfg: dict, strict: bool = False) -> tuple[dict, list[str]]:
    import torch
    from fcl.train_auc_ce import _label_tensor
    if not job["cache"].is_file():
        raise FileNotFoundError(f"Missing {job['source']} feature/pseudo-label cache: {job['cache']}")
    cache = torch_load(job["cache"])
    if not isinstance(cache, dict):
        raise ValueError("Feature cache must be a dictionary")
    required = [f"{split}_{kind}" for split in ("train", "val", "test") for kind in ("feats", "labels")]
    for key in required:
        if key not in cache:
            raise KeyError(f"{job['cache']} has no {key}")
    prior = cache.get("clip_probs_train")
    if not isinstance(prior, torch.Tensor) or prior.ndim != 2 or prior.shape[1] == 0:
        raise ValueError("Training requires a source probability matrix clip_probs_train")
    classes = prior.shape[1]
    warnings = []
    source_splits = prediction_splits(job, cache)
    if job["source"] == "qwen":
        required_source_split = "test" if job["method"] == "zero_shot" else "train"
        if qwen_split_metadata(cache, required_source_split) is None:
            raise ValueError(f"Qwen {job['method']} requires explicit Qwen source metadata for "
                             f"{required_source_split} with replaced_clip_fields=true; CLIP fallback is forbidden")
        for split in source_splits:
            metadata = qwen_split_metadata(cache, split)
            variant = metadata.get("prompt_variant")
            if variant in ("A", "B", "C") and variant != job["prompt"]:
                raise ValueError(f"Qwen {split} prompt metadata {variant} does not match requested {job['prompt']}")
            if variant not in ("A", "B", "C"):
                warnings.append(f"Qwen {split} cache has no verified prompt variant; prompt A/B/C alignment is unverified")
    elif any(qwen_split_metadata(cache, split) for split in source_splits):
        raise ValueError("Selected CLIP source contains Qwen-overlaid prediction fields")
    for split in ("train", "val", "test"):
        features, labels = cache[f"{split}_feats"], cache[f"{split}_labels"]
        if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError(f"{split} features and labels must be tensors")
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels) or not len(labels):
            raise ValueError(f"Invalid/empty {split} features or labels")
        if not torch.isfinite(features).all() or labels.min() < 0 or labels.max() >= classes:
            raise ValueError(f"Invalid {split} values or unavailable ground-truth labels")
        _label_tensor(labels, f"{split}_labels", len(features), classes)
        if features.shape[1] != cache["train_feats"].shape[1] or features.shape[1] == 0:
            raise ValueError(f"{split} feature dimension differs from train or is empty")
        if split not in source_splits:
            continue
        probabilities = cache.get(f"clip_probs_{split}")
        predictions = cache.get(f"clip_argmax_{split}")
        if predictions is not None:
            _label_tensor(predictions, f"clip_argmax_{split}", len(labels), classes)
        if probabilities is not None:
            if not isinstance(probabilities, torch.Tensor) or probabilities.shape != (len(labels), classes) or not torch.isfinite(probabilities).all():
                raise ValueError(f"Invalid {split} source probabilities")
            if probabilities.min() < 0 or not torch.allclose(probabilities.float().sum(1), torch.ones(len(labels)), atol=1e-3):
                raise ValueError(f"{split} source probability rows must sum to one")
            if predictions is not None and not torch.equal(predictions.long().view(-1), probabilities.argmax(1)):
                raise ValueError(f"{split} argmax/probability disagreement; refusing an ambiguous label source")
        elif predictions is None:
            raise ValueError(f"Missing {split} source predictions")
    expected = cfg["datasets"][job["dataset"]]
    if len(cache["train_labels"]) != expected["paper_train_samples"] or classes != expected["paper_classes"]:
        message = (f"Split differs from Table 1: train={len(cache['train_labels'])}, classes={classes}; "
                   f"reference={expected['paper_train_samples']}/{expected['paper_classes']}")
        if strict:
            raise ValueError(message)
        warnings.append(message)
    if cache["train_feats"].shape[1] != 768:
        warnings.append("Feature dimension differs from the paper's CLIP ViT-L/14 dimension 768")
    if job["method"] not in ("zero_shot", "fs_c") and job["obs"].exists():
        observed = torch_load(job["obs"])
        n = len(cache["train_labels"])
        if not isinstance(observed, dict):
            raise ValueError("Observed-label cache must be a dictionary")
        if any(key not in observed or observed[key].numel() != n for key in ("y_obs", "s")):
            raise ValueError(f"Observed-label length mismatch: {job['obs']}")
        if not torch.all((observed["s"] == 0) | (observed["s"] == 1)):
            raise ValueError("Observed flags must be zero (corrected) or one (uncorrected)")
        _label_tensor(observed["y_obs"], "y_obs", n, classes)
        _label_tensor(observed["s"], "s", n, 2)
        meta = observed.get("meta", {})
        if job["source"] == "qwen" and meta.get("label_source") != "qwen":
            raise ValueError("Qwen observed labels require explicit label_source=qwen metadata")
        if meta.get("label_source", job["source"]) != job["source"]:
            raise ValueError("Observed-label source metadata does not match selected source")
        actual_rho = meta.get("top_outlier_pct")
        if actual_rho is not None and abs(float(actual_rho) - job["rho"]) > 1e-8:
            raise ValueError("Observed-label rho metadata does not match requested rho")
    elif job["method"] not in ("zero_shot", "fs_c"):
        warnings.append(f"Observed labels missing: {job['obs']}; use --prepare-labels when executing")
    return cache, warnings


def prepare_observed(job: dict, cache: dict) -> None:
    import torch
    from utils.labels.make_obs_labels_cifar100 import make_observed_labels
    probabilities = cache["clip_probs_train"]
    pseudo = cache.get("clip_argmax_train", probabilities.argmax(1)).long()
    observed, flags = make_observed_labels(cache["train_feats"], cache["train_labels"], pseudo, job["rho"])
    job["obs"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({"y_obs": observed, "s": flags, "meta": {
        "from_cache": str(job["cache"].resolve()), "label_source": job["source"],
        "prompt": job["prompt"], "top_outlier_pct": job["rho"],
        "correction": "oracle simulation using dataset ground truth only at selected indices",
    }}, job["obs"])


def run_ce_or_zeroshot(job: dict, cfg: dict, cache: dict) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from contrast.cache_experiment import evaluate, save_history
    from fcl.train_auc_ce import LinearHead, set_seed
    output, method = job["run_dir"], job["method"]
    if method == "zero_shot":
        metrics = {}
        splits = prediction_splits(job, cache)
        if job["source"] == "qwen" and "test" not in splits:
            raise ValueError("Qwen zero_shot requires Qwen test predictions; CLIP fallback is forbidden")
        for split in splits:
            predictions = cache.get(f"clip_argmax_{split}")
            if predictions is None:
                predictions = cache[f"clip_probs_{split}"].argmax(1)
            metrics[split] = {"accuracy": float(predictions.eq(cache[f"{split}_labels"]).float().mean()),
                              "n": int(predictions.numel())}
        (output / "metrics.json").write_text(json.dumps({"method": method, "source": job["source"],
            "measured": True, "metrics": metrics}, indent=2), encoding="utf-8")
        return
    set_seed(job["seed"])
    train = F.normalize(cache["train_feats"].float(), dim=-1)
    if method == "fs_c":
        target, selected = cache["train_labels"].long(), torch.ones(len(train), dtype=torch.bool)
    else:
        observed = torch_load(job["obs"])
        target = observed["y_obs"].long()
        selected = observed["s"].eq(0 if method == "fl_c" else 1)
    if not selected.any():
        raise ValueError(f"{method}: no selected training samples")
    loader = DataLoader(TensorDataset(train[selected], target[selected]),
                        batch_size=cfg["protocol"]["batch_size"], shuffle=True,
                        num_workers=job.get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearHead(train.shape[1], cache["clip_probs_train"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["protocol"]["lr"],
                                  weight_decay=cfg["protocol"]["weight_decay"])
    history, best, best_epoch = [], -1.0, 0
    val = F.normalize(cache["val_feats"].float(), dim=-1)
    for epoch in range(1, job["epochs"] + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            n += len(y)
        validation = evaluate(model, val, cache["val_labels"], device)
        accuracy = validation["accuracy"]
        history.append({"epoch": epoch, "train_loss": total_loss / n, "val_acc": accuracy,
                        "val_loss": validation["nll"]})
        print(f"[{method}][{job['dataset']}] epoch={epoch} val_acc={accuracy:.6f}", flush=True)
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            torch.save(model.state_dict(), output / "best.ckpt")
    torch.save(model.state_dict(), output / "last.ckpt")
    model.load_state_dict(torch_load(output / "best.ckpt"))
    test = evaluate(model, F.normalize(cache["test_feats"].float(), dim=-1), cache["test_labels"], device)
    save_history(output, history)
    (output / "metrics.json").write_text(json.dumps({"method": method, "source": job["source"],
        "measured": True, "best_val_accuracy": best, "best_epoch": best_epoch,
        "test": test, "selected_train_samples": int(selected.sum()),
        "epochs_completed": job["epochs"]}, indent=2), encoding="utf-8")


def main() -> int:
    # Subprocess logs use UTF-8 on every host. Windows otherwise defaults to
    # GBK for redirected Python output, which can corrupt decoded text or make
    # forwarding a child log raise UnicodeEncodeError and abort a valid run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = arguments()
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.P is not None and args.P <= 0:
        raise ValueError("--P must be positive")
    if args.K is not None and args.K <= 0:
        raise ValueError("--K must be positive")
    if args.pairwise_tau is not None and args.pairwise_tau <= 0:
        raise ValueError("--pairwise-tau must be positive")
    cfg = load_yaml(args.config)
    jobs = build_jobs(cfg, args)
    if not jobs:
        raise ValueError("No jobs selected")
    print(f"[PLAN] {len(jobs)} jobs; mode={'execute' if args.execute else 'preflight' if args.preflight else 'dry-run'}")
    errors, audited = [], set()
    source_sha, git_dirty = source_fingerprint() if (args.preflight or args.execute) else (None, None)
    for job in jobs:
        command = command_for(job, cfg)
        print(f"[{job['suite']}] {job['dataset']}/{job['source']}/{job['method']} "
              f"prompt={job['prompt']} rho={job['rho']} lam={job['lam']} seed={job['seed']}")
        print(shlex.join(command) if command else f"  internal exact {job['method']} runner; cache={job['cache']}; obs={job['obs']}")
        if not (args.preflight or args.execute):
            continue
        manifest_path = None
        metadata = None
        try:
            key = (job["cache"], job["obs"], job["method"] in ("zero_shot", "fs_c"))
            cache, warnings = inspect_input(job, cfg, args.strict_paper_splits)
            if key not in audited:
                for warning in warnings:
                    print(f"[AUDIT] {warning}")
                audited.add(key)
            if not args.execute:
                if job["method"] not in ("zero_shot", "fs_c") and not job["obs"].exists() and not args.prepare_labels:
                    raise FileNotFoundError(f"Missing observed labels: {job['obs']}")
                continue
            if job["method"] not in ("zero_shot", "fs_c") and not job["obs"].exists():
                if not args.prepare_labels:
                    raise FileNotFoundError(f"Missing observed labels: {job['obs']}; pass --prepare-labels")
                prepare_observed(job, cache)
                # Validate the generated cache before including it in the run fingerprint.
                cache, warnings = inspect_input(job, cfg, args.strict_paper_splits)
            fingerprint, fingerprints = job_fingerprint(job, source_sha or "", git_dirty)
            identity = job_identity(job)
            if args.resume:
                previous = find_completed_job(args.output_root, identity, fingerprint)
                if previous is not None:
                    print(f"[RESUME] skip completed job at {previous}", flush=True)
                    continue
            output = job["run_dir"]
            (output / "logs").mkdir(parents=True, exist_ok=True)
            manifest_path = output / "paper_run.json"
            metadata = {**job, "config": cfg, "command": command, "status": "running",
                        "reference_targets": "configs/paper_targets.yaml", "measured": False,
                        "job_key": identity, "fingerprint": fingerprint,
                        "fingerprints": fingerprints, "source_sha256": source_sha,
                        "cache_sha256": fingerprints["cache_sha256"],
                        "obs_sha256": fingerprints["obs_sha256"],
                        "git_dirty": git_dirty, "gitdirty": git_dirty,
                        "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
            manifest_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            if command:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                with (output / "logs" / "console.log").open("w", encoding="utf-8") as log:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    code = proc.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
                if job["method"] in ("l2b", "dld", "csgn"):
                    from scripts.run_legacy_baselines import finalize
                    finalizer_args = argparse.Namespace(**{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    })
                    finalize(output, job["method"], job["dataset"], job["cache"], command,
                             finalizer_args, dt.datetime.now(dt.timezone.utc).isoformat(), "completed")
            else:
                run_ce_or_zeroshot(job, cfg, cache)
            metadata.update(status="completed", measured=True,
                            ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
            manifest_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            print(f"[ERROR] {job['dataset']}/{job['method']}: {exc}", file=sys.stderr, flush=True)
            if metadata is not None and manifest_path is not None:
                metadata.update(status="failed", measured=False, error=str(exc),
                                ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
                manifest_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            elif args.execute:
                # Input/audit failures happen before the normal running manifest
                # is created; persist those failures as well for later review.
                output = job["run_dir"]
                output.mkdir(parents=True, exist_ok=True)
                failure_fingerprint, failure_files = job_fingerprint(job, source_sha or "", git_dirty)
                fallback = {**job, "config": cfg, "command": command, "status": "failed",
                             "measured": False, "error": str(exc),
                             "job_key": job_identity(job), "fingerprint": failure_fingerprint,
                             "fingerprints": failure_files,
                             "cache_sha256": failure_files["cache_sha256"],
                             "obs_sha256": failure_files["obs_sha256"],
                             "source_sha256": source_sha, "git_dirty": git_dirty,
                             "gitdirty": git_dirty,
                             "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                             "ended_at": dt.datetime.now(dt.timezone.utc).isoformat()}
                (output / "paper_run.json").write_text(json.dumps(fallback, indent=2, default=str), encoding="utf-8")
            errors.append(str(exc))
            if not args.continue_on_error:
                return 1
    print(f"[DONE] jobs={len(jobs)} errors={len(errors)}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
