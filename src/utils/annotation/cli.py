"""Import legacy labels, run Qwen, or overlay identity-aligned labels on a cache."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .schema import AnnotationError, Manifest, fingerprint, make_record, parse_label, read_classes
from .store import AnnotationStore, load_annotations


def import_labels(args):
    manifest, classes = Manifest.read(args.manifest, args.split), read_classes(args.classes)
    source = Path(args.input)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    if any(not line.strip() for line in lines):
        raise AnnotationError("Empty legacy rows are rejected because they can shift sample alignment")
    ordered = args.ordered_legacy
    if ordered and len(lines) != len(manifest.samples):
        raise AnnotationError(f"Ordered legacy input length {len(lines)} != manifest length {len(manifest.samples)}")
    pending, seen = [], set()
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line) if line.lstrip().startswith("{") else None
            if raw is None:
                if "\t" in line:
                    identity, label = line.rsplit("\t", 1)
                    raw = {"sample_id": identity, "label": label} if identity in manifest.by_id else {"path": identity, "label": label}
                elif ordered:
                    raw = {**manifest.samples[index], "label": line}
                else:
                    raise AnnotationError("Bare legacy label rows require --ordered-legacy and an exact original-order manifest")
            if not isinstance(raw, dict):
                raise AnnotationError("JSONL row must be an object")
            sample = manifest.match(raw)
            labels = {key: raw[key] for key in ("label_id", "label", "prediction", "class_name") if key in raw}
            # Unified records contain both numeric label and class_name; validate both.
            if "label_id" in labels and "class_name" in labels:
                label = parse_label(labels["label_id"], classes, args.label_base)
                if parse_label(labels["class_name"], classes) != label:
                    raise AnnotationError("label_id and class_name disagree")
            else:
                label = parse_label(labels, classes, args.label_base)
            key = sample["sample_id"]
            if key in seen:
                raise AnnotationError(f"Duplicate annotation identity: {key}")
            seen.add(key)
            pending.append(make_record(sample, label, classes, args.source_model,
                                       raw.get("probabilities", raw.get("probs")), line,
                                       "explicit_legacy_order" if ordered else "identity"))
        except (ValueError, KeyError, TypeError) as exc:
            raise AnnotationError(f"Rejected input line {index+1}: {exc}") from exc
    if not args.allow_partial and len(seen) != len(manifest.samples):
        raise AnnotationError(f"Only {len(seen)}/{len(manifest.samples)} samples present; use --allow-partial to import a subset")
    settings = {"mode": "import", "input_sha256": fingerprint(lines), "label_base": args.label_base,
                "ordered_legacy": ordered}
    store = AnnotationStore(args.output, manifest, classes, args.source_model, settings, args.resume)
    for record in pending:
        existing = store.records.get(record["sample_id"])
        if existing:
            if existing != record:
                raise AnnotationError(f"Resumed label differs for {record['sample_id']}")
            continue
        store.append(record)
    print(json.dumps({"output": str(store.path), "annotated": len(store.records),
                      "expected": len(manifest.samples), "one_hot_is_not_calibrated_probability": True}))


def build_prompt(classes, variant="C", template_path=None):
    candidates = "\n".join(f"{index}: {name}" for index, name in enumerate(classes))
    if template_path:
        template = Path(template_path).read_text(encoding="utf-8")
        if "{classes}" not in template:
            raise AnnotationError("Custom prompt must contain {classes}")
        return template.replace("{classes}", candidates)
    templates = {
        "A": "请根据图片从以下候选类别中选择唯一最合适的类别。仅输出零起始类别编号的 JSON 对象，格式为 {{\"label_id\": 0}}，不要解释。\n{classes}",
        "B": "Choose the single candidate category that best matches this image. Return only a JSON object with its zero-based category ID, such as {{\"label_id\": 0}}. Do not explain.\n{classes}",
        "C": "Classify the image. Return only {{\"label_id\": ID}} using one zero-based candidate ID.\n{classes}",
    }
    return templates[variant].format(classes=candidates)


class QwenPredictor:
    """Two explicit backends; the historical Qwen-VL and Qwen2.5-VL are distinct models."""
    def __init__(self, model_name, backend, device_map="auto", trust_remote_code=False):
        import torch
        self.backend = backend
        if backend == "qwen2.5-vl":
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map=device_map)
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            if not trust_remote_code:
                raise AnnotationError("Historical Qwen-VL-Chat requires explicit --trust-remote-code")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map=device_map, trust_remote_code=True)
        self.model.eval()
        self.torch = torch

    def __call__(self, path, prompt, max_new_tokens):
        with self.torch.inference_mode():
            if self.backend == "qwen-vl-chat":
                query = self.processor.from_list_format([{"image": str(path)}, {"text": prompt}])
                answer, _ = self.model.chat(self.processor, query=query, history=None,
                                            do_sample=False, max_new_tokens=max_new_tokens)
                return answer
            from PIL import Image
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                        {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0].strip()


def annotate(args):
    manifest, classes = Manifest.read(args.manifest, args.split), read_classes(args.classes)
    prompt = build_prompt(classes, args.prompt_variant, args.prompt_file)
    settings = {"mode": "qwen", "backend": args.backend, "prompt_sha256": fingerprint(prompt),
                "prompt_variant": "custom" if args.prompt_file else args.prompt_variant,
                "max_new_tokens": args.max_new_tokens, "image_root": str(Path(args.image_root).resolve())}
    meta_path = Path(args.output).with_suffix(Path(args.output).suffix + ".meta.json")
    if args.resume and meta_path.exists():
        previous_settings = json.loads(meta_path.read_text(encoding="utf-8")).get("settings", {})
        if "prompt_variant" not in previous_settings:
            # Older progress still verifies the exact prompt hash; adding a
            # descriptive variant must not prevent a safe interrupted resume.
            settings.pop("prompt_variant")
    store = AnnotationStore(args.output, manifest, classes, args.model, settings, args.resume)
    todo = [sample for sample in manifest.samples if sample["sample_id"] not in store.records]
    if args.limit is not None:
        todo = todo[:args.limit]
    if not todo:
        print(f"No pending samples ({len(store.records)}/{len(manifest.samples)})")
        return
    predictor = QwenPredictor(args.model, args.backend, args.device_map, args.trust_remote_code)
    errors = 0
    for sample in todo:
        path = Path(sample["path"])
        if not path.is_absolute():
            path = Path(args.image_root) / path
        raw_output = None
        try:
            if not sample["path"] or not path.is_file():
                raise AnnotationError(f"Image not found: {path}")
            raw_output = predictor(path, prompt, args.max_new_tokens)
            label = parse_label(raw_output, classes)
            record = make_record(sample, label, classes, args.model, raw_output=raw_output)
            store.append(record)
            print(f"[{len(store.records)}/{len(manifest.samples)}] {sample['sample_id']} -> {label}", flush=True)
        except Exception as exc:
            errors += 1
            with Path(str(args.output) + ".errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({**sample, "error": str(exc), "raw_output": raw_output}, ensure_ascii=False) + "\n")
            if not args.continue_on_error:
                raise AnnotationError(f"Rejected sample {sample['sample_id']}; progress saved, use --resume: {exc}") from exc
    print(json.dumps({"annotated": len(store.records), "expected": len(manifest.samples), "errors": errors}))
    if errors:
        raise AnnotationError("Some samples failed; successful progress is saved, rerun with --resume")


def export_cache(args):
    import torch
    manifest, classes = Manifest.read(args.manifest, args.split), read_classes(args.classes)
    records = load_annotations(args.annotations, manifest, classes)
    if len(records) != len(manifest.samples):
        raise AnnotationError("Cache export requires labels for every sample in the selected manifest split")
    # Feature caches are data artifacts.  Keep pickle execution disabled when
    # the runtime supports the safer loader; old PyTorch versions are handled
    # by the narrow compatibility fallback.
    try:
        cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(args.cache, map_location="cpu")
    split = args.split or "train"
    cached_names = cache.get("class_names", cache.get("classnames", cache.get("meta", {}).get("classes")))
    if cached_names is not None:
        cached_names = [name.decode() if isinstance(name, bytes) else str(name) for name in cached_names]
        duplicates = {name.casefold() for name in cached_names
                      if sum(other.casefold() == name.casefold() for other in cached_names) > 1}
        display_names = [f"{name} ({cache['wnids'][i]})" if name.casefold() in duplicates and "wnids" in cache
                         else name for i, name in enumerate(cached_names)]
        if classes != display_names:
            raise AnnotationError("Cache class order differs from annotation class order")
    if f"clip_probs_{split}" in cache and cache[f"clip_probs_{split}"].shape[1] != len(classes):
        raise AnnotationError("Cache class count differs from annotation class count")
    count = len(cache[f"{split}_feats"])
    if count != len(manifest.samples):
        raise AnnotationError(f"Cache has {count} samples but manifest has {len(manifest.samples)}")
    id_keys = (f"{split}_sample_ids", f"sample_ids_{split}")
    path_keys = (f"{split}_paths", f"paths_{split}")
    identity = next((cache[k] for k in id_keys if k in cache), None)
    path_identity = next((cache[k] for k in path_keys if k in cache), None)
    if identity is None and isinstance(cache.get("sample_ids"), dict):
        identity = cache["sample_ids"].get(split)
    if path_identity is None and isinstance(cache.get("sample_paths"), dict):
        path_identity = cache["sample_paths"].get(split)
    if identity is not None:
        ordered = [manifest.match({"sample_id": str(value)}) for value in identity]
    elif path_identity is not None:
        ordered = [manifest.match({"path": str(value)}) for value in path_identity]
    elif args.confirm_cache_manifest_order:
        # Old caches omitted paths; caller must explicitly supply their original extraction order.
        ordered = manifest.samples
    else:
        raise AnnotationError("Cache lacks sample identities; --confirm-cache-manifest-order is required for a legacy cache")
    if len({sample["sample_id"] for sample in ordered}) != count:
        raise AnnotationError("Cache sample identities are duplicated")
    labels = [records[row["sample_id"]] for row in ordered]
    probabilities = torch.tensor([row["probabilities"] for row in labels], dtype=torch.float32)
    argmax = torch.tensor([row["label_id"] for row in labels], dtype=torch.long)
    cache[f"vlm_probs_{split}"] = probabilities
    cache[f"vlm_argmax_{split}"] = argmax
    if args.replace_clip_fields:
        # Keep a recoverable CLIP prior when the training CLI expects clip_* names.
        for key in (f"clip_probs_{split}", f"clip_argmax_{split}"):
            if key in cache:
                cache.setdefault("original_" + key, cache[key])
        cache[f"clip_probs_{split}"], cache[f"clip_argmax_{split}"] = probabilities, argmax
    cache[f"{split}_sample_ids"] = [row["sample_id"] for row in ordered]
    cache[f"{split}_paths"] = [row["path"] for row in ordered]
    source_models = sorted({row["source_model"] for row in labels})
    annotation_meta_path = Path(args.annotations).with_suffix(Path(args.annotations).suffix + ".meta.json")
    annotation_settings = {}
    if annotation_meta_path.exists():
        annotation_settings = json.loads(annotation_meta_path.read_text(encoding="utf-8")).get("settings", {})
    cache.setdefault("annotation_metadata", {})[split] = {
        "manifest_sha256": manifest.sha256, "annotations_sha256": fingerprint(labels),
        "source_models": source_models,
        "probability_sources": sorted({row["probability_source"] for row in labels}),
        "prompt_variant": annotation_settings.get("prompt_variant"),
        "prompt_sha256": annotation_settings.get("prompt_sha256"),
        "legacy_cache_order_confirmed": identity is None and path_identity is None,
        "replaced_clip_fields": args.replace_clip_fields}
    output = Path(args.output)
    if output.exists() or output.resolve() == Path(args.cache).resolve():
        raise AnnotationError("Use a new output cache path; input cache is read-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if temp.exists():
        raise AnnotationError(f"Temporary output exists: {temp}")
    torch.save(cache, temp)
    os.replace(temp, output)
    print(f"Saved {count} identity-aligned VLM labels to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    def inputs(command):
        command.add_argument("--manifest", required=True, help="JSONL/JSON/CSV sample_id,path,split")
        command.add_argument("--classes", required=True, help="Ordered JSON list or one class per line")
        command.add_argument("--split", help="Select this manifest split")
        command.add_argument("--output", required=True)
    importer = sub.add_parser("import", help="Import old *_label_pre.txt or identity-keyed JSONL/TSV")
    inputs(importer)
    importer.add_argument("--input", required=True)
    importer.add_argument("--source-model", required=True, help="Record the exact original model; never infer from filename")
    importer.add_argument("--label-base", type=int, choices=(0, 1), default=0)
    importer.add_argument("--ordered-legacy", action="store_true", help="Explicitly assert manifest is the exact original label-file order")
    importer.add_argument("--allow-partial", action="store_true")
    importer.add_argument("--resume", action="store_true")
    importer.set_defaults(func=import_labels)
    qwen = sub.add_parser("qwen", help="Run optional local transformers VLM inference")
    inputs(qwen)
    qwen.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    qwen.add_argument("--backend", choices=("qwen2.5-vl", "qwen-vl-chat"), default="qwen2.5-vl")
    qwen.add_argument("--image-root", default=".")
    qwen.add_argument("--device-map", default="auto")
    qwen.add_argument("--prompt-variant", choices=("A", "B", "C"), default="C")
    qwen.add_argument("--prompt-file", help="Exact historical prompt template, containing {classes}")
    qwen.add_argument("--max-new-tokens", type=int, default=32)
    qwen.add_argument("--trust-remote-code", action="store_true")
    qwen.add_argument("--continue-on-error", action="store_true")
    qwen.add_argument("--limit", type=int)
    qwen.add_argument("--resume", action="store_true")
    qwen.set_defaults(func=annotate)
    exporter = sub.add_parser("export-cache", help="Add VLM labels to a new feature cache")
    inputs(exporter)
    exporter.add_argument("--cache", required=True)
    exporter.add_argument("--annotations", required=True)
    exporter.add_argument("--confirm-cache-manifest-order", action="store_true")
    exporter.add_argument("--replace-clip-fields", action="store_true", help="Explicit compatibility mode for old train scripts")
    exporter.set_defaults(func=export_cache)
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if getattr(args, "max_new_tokens", 1) <= 0:
        parser.error("--max-new-tokens must be positive")
    try:
        args.func(args)
    except AnnotationError as exc:
        parser.exit(2, f"Annotation error: {exc}\n")


if __name__ == "__main__":
    main()
