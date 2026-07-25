"""Centralized project paths.

Environment variables take precedence over ``configs/paths.yaml``:

    DATA_ROOT=/mnt/data OUTPUT_ROOT=/mnt/experiments python src/fcl/train_auc_ce.py

Relative environment values are resolved against the project root.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "paths.yaml"


def _read_defaults() -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:  # Keep path resolution usable before optional deps are installed.
        values: Dict[str, Any] = {}
        for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip("'\"")
        return values
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


_DEFAULTS = _read_defaults()


def _rooted(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


DATA_ROOT = _rooted(os.environ.get("DATA_ROOT", _DEFAULTS.get("data_root", "./data")))
OUTPUT_ROOT = _rooted(os.environ.get("OUTPUT_ROOT", _DEFAULTS.get("output_root", "./outputs")))
WEIGHTS_ROOT = _rooted(os.environ.get("WEIGHTS_ROOT", _DEFAULTS.get("weights_root", "./weights")))
RAW_ROOT = DATA_ROOT / _DEFAULTS.get("raw_dirname", "raw")
PROCESSED_ROOT = DATA_ROOT / _DEFAULTS.get("processed_dirname", "processed")
SHARED_CACHE_ROOT = OUTPUT_ROOT / _DEFAULTS.get("shared_cache_dirname", "cache")


def dataset_root(dataset: str) -> Path:
    """Return the read-only raw dataset directory."""
    return RAW_ROOT / dataset


def clip_cache(dataset: str, model: str = "ViT-L-14") -> Path:
    """Return the reusable CLIP feature cache for a dataset."""
    slug = dataset.lower().replace("-", "").replace("_", "")
    model_slug = model.lower().replace("/", "-").replace(".", "")
    return PROCESSED_ROOT / f"{slug}_clip_{model_slug}_embeddings.pt"


def observed_labels_cache(dataset: str) -> Path:
    slug = dataset.lower().replace("-", "").replace("_", "")
    return PROCESSED_ROOT / f"{slug}_observed_labels.pt"


def legacy_label_path(dataset: str, source: str, filename: str) -> Path:
    """Writable destination for legacy CLIP/Qwen label-generation outputs."""
    slug = dataset.lower().replace("-", "").replace("_", "")
    return PROCESSED_ROOT / "legacy_labels" / slug / source / filename


def experiment_root(name: str) -> Path:
    return OUTPUT_ROOT / name


def experiment_checkpoint(name: str, filename: str = "best.ckpt") -> Path:
    return experiment_root(name) / filename


def experiment_logs(name: str) -> Path:
    return experiment_root(name) / "logs"


def clip_weights(filename: str = "clip_vit_b32.pt") -> Path:
    return SHARED_CACHE_ROOT / filename


def save_experiment_config(name: str, args: Any) -> Path:
    """Snapshot CLI arguments and resolved roots into the ignored output dir."""
    destination = experiment_root(name) / "config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": name,
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
    }
    try:
        import yaml
        destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except ImportError:
        # Valid, dependency-free fallback for environments bootstrapping PyYAML.
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
