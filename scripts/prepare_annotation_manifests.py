"""Export label-free image manifests in the exact order of a frozen CLIP cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import PROCESSED_ROOT, RAW_ROOT
from utils.annotation.schema import Manifest, fingerprint


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to replace a different manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def cifar_images(cache, raw_root, image_root):
    """Read only the trusted official CIFAR Python archive and verify each label."""
    import numpy as np
    import torch
    from PIL import Image
    with (Path(raw_root) / "meta").open("rb") as handle:
        names = pickle.load(handle, encoding="latin1")["fine_label_names"]
    names = [name.decode() if isinstance(name, bytes) else str(name) for name in names]
    if list(cache.get("class_names", cache.get("classnames", []))) != names:
        raise ValueError("CIFAR official/cache class order differs")
    originals = {}
    for name in ("train", "test"):
        with (Path(raw_root) / name).open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        originals[name] = (data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
                           torch.tensor(data["fine_labels"]))
    result, identities = {}, {}
    for split in ("train", "val", "test"):
        official = "test" if split == "test" else "train"
        images, labels = originals[official]
        key = "train_idx_from_train50k" if split == "train" else "val_idx_from_train50k"
        raw_indices = torch.arange(len(images)) if split == "test" else cache[key]
        if (not isinstance(raw_indices, torch.Tensor) or raw_indices.ndim != 1
                or raw_indices.is_floating_point() or raw_indices.is_complex() or raw_indices.dtype == torch.bool
                or (raw_indices < 0).any() or (raw_indices >= len(images)).any()):
            raise ValueError(f"CIFAR saved indices invalid: {split}")
        indices = raw_indices.long()
        if not torch.equal(labels[indices], cache[f"{split}_labels"].long()):
            raise ValueError(f"CIFAR official/cache label alignment failed: {split}")
        expected_ids = [f"{'test10k' if split == 'test' else 'train50k'}/{int(i):05d}" for i in indices]
        saved_ids = cache.get(f"{split}_sample_ids", cache.get("sample_ids", {}).get(split))
        if saved_ids is not None and saved_ids != expected_ids:
            raise ValueError(f"CIFAR sample IDs do not match saved split indices: {split}")
        identities[split] = expected_ids
        result[split] = []
        for sample_id, index in zip(expected_ids, indices.tolist()):
            destination = Path(image_root) / (sample_id + ".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                with Image.open(destination) as old:
                    if not np.array_equal(np.asarray(old.convert("RGB")), images[index]):
                        raise ValueError(f"CIFAR exported image differs: {destination}")
            else:
                Image.fromarray(images[index]).save(destination)
            result[split].append(str(destination.resolve()))
    return result, identities


def recover_indexed_images(dataset, cache, raw_root):
    """Replay the original torchvision loader and the saved, never resampled indices."""
    import torch
    import torchvision
    if dataset == "caltech101":
        loader_root = next((root for root in (raw_root, raw_root / "Caltech101")
                            if (root / "caltech101/101_ObjectCategories").is_dir()), None)
        if loader_root is None:
            raise FileNotFoundError(f"No Caltech101 torchvision data under {raw_root}")
        base = torchvision.datasets.Caltech101(root=loader_root, download=False)
        classes = [name for name in base.categories if name != "BACKGROUND_Google"]
        records = [(Path(base.root) / "101_ObjectCategories" / base.categories[int(label)] /
                    f"image_{int(index):04d}.jpg", classes.index(base.categories[int(label)]))
                   for label, index in zip(base.y, base.index)
                   if base.categories[int(label)] != "BACKGROUND_Google"]
    elif dataset == "eurosat":
        loader_root = next((root for root in (raw_root, raw_root / "EuroSAT")
                            if (root / "eurosat/2750").is_dir()), None)
        if loader_root is None:
            raise FileNotFoundError(f"No EuroSAT torchvision data under {raw_root}")
        base = torchvision.datasets.EuroSAT(root=loader_root, download=False)
        classes, records = base.classes, base.samples
    else:
        raise ValueError(f"No audited indexed identity recovery for {dataset}")
    if list(cache["class_names"]) != list(classes):
        raise ValueError(f"Recovered loader class order differs from frozen cache: {dataset}")
    all_targets = torch.tensor([label for _, label in records])
    paths, ids, index_hashes = {}, {}, {}
    seen = set()
    for split in ("train", "val", "test"):
        indices = cache[f"{split}_indices"]
        if (not isinstance(indices, torch.Tensor) or indices.ndim != 1
                or indices.is_floating_point() or indices.is_complex() or indices.dtype == torch.bool):
            raise ValueError(f"Invalid saved {dataset}/{split} indices")
        indices = indices.long()
        if len(indices) != len(cache[f"{split}_labels"]) or (indices < 0).any() or (indices >= len(records)).any():
            raise ValueError(f"Saved {dataset}/{split} indices are out of range or misaligned")
        if not torch.equal(all_targets[indices], cache[f"{split}_labels"].long()):
            raise ValueError(f"Recovered images/labels disagree with frozen cache: {dataset}/{split}")
        values = indices.tolist()
        if len(set(values)) != len(values) or seen.intersection(values):
            raise ValueError(f"Saved {dataset} split indices overlap or contain duplicates")
        seen.update(values)
        paths[split] = [str(Path(records[index][0]).resolve()) for index in values]
        ids[split] = [f"{dataset}/{index:05d}" for index in values]
        index_hashes[split] = fingerprint(values)
    return paths, ids, {"method": "original_torchvision_loader_and_saved_split_indices",
                         "torchvision": torchvision.__version__, "split_index_sha256": index_hashes,
                         "class_order_and_all_labels_verified": True}


def consistent_identity(cache, split, kind, recovered=None):
    """Every saved alias must agree; a recovered mapping cannot hide conflicts."""
    if kind == "id":
        keys, nested = (f"{split}_sample_ids", f"sample_ids_{split}"), "sample_ids"
        normalize = str
    else:
        keys, nested = (f"{split}_paths", f"paths_{split}"), "sample_paths"
        normalize = lambda value: str(Path(value).resolve())
    candidates = [(key, cache[key]) for key in keys if key in cache]
    if isinstance(cache.get(nested), dict) and split in cache[nested]:
        candidates.append((f"{nested}.{split}", cache[nested][split]))
    expected = [normalize(value) for value in recovered] if recovered is not None else None
    for name, values in candidates:
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Cache identity alias {name} must be a list")
        current = [normalize(value) for value in values]
        if expected is not None and expected != current:
            raise ValueError(f"Cache identity alias disagrees with verified order: {name}")
        expected = current
    return expected


def prepare(dataset, cache_path, destination, cifar_raw=None, raw_root=None):
    import torch
    cache_path, destination = Path(cache_path), Path(destination)
    raw_root = Path(raw_root or RAW_ROOT)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    classes = cache.get("class_names", cache.get("classnames", cache.get("meta", {}).get("classes")))
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("Cache has no ordered class vocabulary")
    classes = [name.decode() if isinstance(name, bytes) else str(name) for name in classes]
    if len(classes) != cache["clip_probs_train"].shape[1]:
        raise ValueError("Class vocabulary and CLIP probability width disagree")
    for split in ("train", "val", "test"):
        labels, features = cache[f"{split}_labels"], cache[f"{split}_feats"]
        if (not isinstance(labels, torch.Tensor) or labels.ndim != 1 or not len(labels)
                or labels.is_complex() or labels.dtype == torch.bool or not torch.isfinite(labels).all()
                or (labels.is_floating_point() and not torch.equal(labels, labels.round()))
                or (labels < 0).any() or (labels >= len(classes)).any()
                or not isinstance(features, torch.Tensor) or features.ndim != 2 or len(features) != len(labels)):
            raise ValueError(f"Invalid frozen features/ground-truth labels: {dataset}/{split}")
    # Label IDs remain unchanged even when the display strings need disambiguation.
    duplicates = {x.casefold() for x in classes if sum(y.casefold() == x.casefold() for y in classes) > 1}
    display_classes = [f"{name} ({cache['wnids'][i]})" if name.casefold() in duplicates and "wnids" in cache else name
                       for i, name in enumerate(classes)]
    if len({x.casefold() for x in display_classes}) != len(display_classes):
        raise ValueError("Ambiguous duplicate class names without stable class IDs")
    paths = cache.get("sample_paths", {})
    recovered_ids, recovery = {}, None
    if dataset == "cifar100":
        official_root = Path(cifar_raw or raw_root / "CIFAR100")
        paths, recovered_ids = cifar_images(cache, official_root,
                                           destination / "images")
        recovery = {"method": "official_cifar_pixels_and_saved_split_indices",
                    "official_files_sha256": {name: sha256_file(official_root / name) for name in ("train", "test", "meta")},
                    "class_order_and_all_labels_verified": True}
    elif dataset in ("caltech101", "eurosat") and not all(paths.get(split) for split in ("train", "val", "test")):
        paths, recovered_ids, recovery = recover_indexed_images(dataset, cache, raw_root)
        recovery["historical_features_reencoded"] = False
        markers = [path for path in (raw_root / ".fcl_extraction.json", raw_root / dataset / ".fcl_extraction.json") if path.is_file()]
        recovery["extraction_records"] = {str(path): {"sha256": sha256_file(path),
                                                      "content": json.loads(path.read_text(encoding="utf-8"))}
                                           for path in markers}
    rows, split_counts = [], {}
    all_ids = set()
    all_paths = set()
    for split in ("train", "val", "test"):
        ids = consistent_identity(cache, split, "id", recovered_ids.get(split))
        split_paths = consistent_identity(cache, split, "path", paths.get(split))
        n = len(cache[f"{split}_labels"])
        if ids is None or split_paths is None or len(ids) != n or len(split_paths) != n:
            raise ValueError(f"Missing/misaligned stable cache identities for {dataset}/{split}")
        for sample_id, image in zip(ids, split_paths):
            image = Path(image).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            sample_id = str(sample_id)
            if sample_id in all_ids or str(image) in all_paths:
                raise ValueError(f"Duplicate sample identity or overlapping splits: {sample_id}")
            all_ids.add(sample_id)
            all_paths.add(str(image))
            rows.append({"sample_id": sample_id, "path": str(image), "split": split})
        split_counts[split] = n
    manifest = Manifest(rows)
    write_immutable(destination / "manifest.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_immutable(destination / "classes.json", json.dumps(display_classes, ensure_ascii=False, indent=2) + "\n")
    metadata = {"dataset": dataset, "cache": str(cache_path.resolve()), "cache_sha256": sha256_file(cache_path),
                "manifest_sha256": manifest.sha256, "class_names_sha256": fingerprint(display_classes),
                "original_class_names": classes, "display_class_names": display_classes,
                "split_counts": split_counts, "order": "exact cache order in every split",
                "identity_recovery": recovery,
                "legacy_cache_order_verified": bool(recovery),
                "ground_truth_in_manifest": False}
    write_immutable(destination / "manifest_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROCESSED_ROOT / "annotation_manifests")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--cifar-raw", type=Path)
    args = parser.parse_args()
    for dataset in args.datasets:
        prepare(dataset, args.processed_root / f"{dataset}_clip_vit-l-14_embeddings.pt",
                args.output_root / dataset, args.cifar_raw, args.raw_root)


if __name__ == "__main__":
    main()
