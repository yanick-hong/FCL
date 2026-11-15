"""Dependency-free annotation schema. Label IDs in saved files are zero based."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import posixpath
import re
from pathlib import Path


class AnnotationError(ValueError):
    pass


def canonical_path(value):
    # Do not resolve against cwd: manifests are portable and may be made on Windows.
    return posixpath.normpath(str(value).replace("\\", "/")) if value else ""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_classes(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix.lower() == ".json":
        value = json.loads(text)
        classes = value.get("class_names", value.get("classes")) if isinstance(value, dict) else value
    else:
        classes = text.splitlines()
    if not isinstance(classes, list) or not classes or any(not isinstance(c, str) or not c.strip() for c in classes):
        raise AnnotationError("Classes must be a nonempty ordered list of nonempty names")
    classes = [c.strip() for c in classes]
    if len({c.casefold() for c in classes}) != len(classes):
        raise AnnotationError("Class names must be unique (case insensitive)")
    return classes


class Manifest:
    """Stable sample identity and ordering, without exposing ground truth to a VLM."""
    def __init__(self, samples):
        self.samples = []
        self.by_id, self.by_path = {}, {}
        for raw in samples:
            if not isinstance(raw, dict):
                raise AnnotationError("Manifest rows must be objects")
            path = canonical_path(raw.get("path", raw.get("image_path", "")))
            sample_id = str(raw.get("sample_id", raw.get("id", path)))
            if not sample_id:
                raise AnnotationError("Each manifest row requires sample_id or path")
            if sample_id in self.by_id or path and path in self.by_path:
                raise AnnotationError(f"Duplicate manifest identity: {sample_id!r} / {path!r}")
            row = {"sample_id": sample_id, "path": path, "split": str(raw.get("split", "train"))}
            self.samples.append(row)
            self.by_id[sample_id] = row
            if path:
                self.by_path[path] = row
        if not self.samples:
            raise AnnotationError("Manifest is empty")
        self.sha256 = fingerprint(self.samples)

    @classmethod
    def read(cls, path, split=None):
        path = Path(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            if path.suffix.lower() == ".csv":
                rows = list(csv.DictReader(handle))
            elif path.suffix.lower() == ".json":
                payload = json.load(handle)
                rows = payload.get("samples") if isinstance(payload, dict) else payload
            else:
                rows = [json.loads(line) for line in handle if line.strip()]
        if not isinstance(rows, list):
            raise AnnotationError("Manifest must contain a sample list")
        if split is not None:
            rows = [row for row in rows if str(row.get("split", "train")) == split]
        return cls(rows)

    def match(self, raw):
        sample_id = raw.get("sample_id", raw.get("id"))
        path = canonical_path(raw.get("path", raw.get("image_path", "")))
        by_id = self.by_id.get(str(sample_id)) if sample_id is not None else None
        by_path = self.by_path.get(path) if path else None
        if sample_id is not None and by_id is None:
            raise AnnotationError(f"Unknown sample_id: {sample_id!r}")
        if path and by_path is None:
            raise AnnotationError(f"Unknown sample path: {path!r}")
        if by_id is not None and by_path is not None and by_id != by_path:
            raise AnnotationError("sample_id and path refer to different manifest samples")
        row = by_id or by_path
        if row is None:
            raise AnnotationError("Annotation needs a manifest sample_id or path")
        return row


def parse_label(value, classes, label_base=0):
    """Accept only an exact class name, integer, or JSON label object; never first-character parsing."""
    if isinstance(value, bool):
        raise AnnotationError("Boolean is not a class ID")
    if isinstance(value, dict):
        candidates = [value[key] for key in ("label_id", "label", "prediction", "class_name") if key in value]
        if len(candidates) != 1:
            raise AnnotationError("Label object must have exactly one label_id/label/prediction/class_name")
        return parse_label(candidates[0], classes, label_base)
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("{"):
            try:
                return parse_label(json.loads(value), classes, label_base)
            except json.JSONDecodeError as exc:
                raise AnnotationError("Malformed JSON label output") from exc
        names = {name.casefold(): i for i, name in enumerate(classes)}
        if value.casefold() in names:
            return names[value.casefold()]
        if not re.fullmatch(r"[+-]?\d+", value):
            raise AnnotationError(f"Output is not one exact class ID or name: {value!r}")
        value = int(value)
    if not isinstance(value, int):
        raise AnnotationError("Class ID must be an integer")
    label = value - label_base
    if not 0 <= label < len(classes):
        raise AnnotationError(f"Class ID {value} outside [{label_base}, {len(classes)-1+label_base}]")
    return label


def make_record(sample, label, classes, model, probabilities=None, raw_output=None,
                alignment="identity", probability_source=None):
    label = parse_label(label, classes)
    if probabilities is None:
        probabilities = [float(i == label) for i in range(len(classes))]
        probability_source = "one_hot_from_label"
    else:
        if not isinstance(probabilities, (list, tuple)) or len(probabilities) != len(classes):
            raise AnnotationError("Probability length does not match class count")
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in probabilities):
            raise AnnotationError("Probabilities must be finite nonnegative numbers")
        total = sum(probabilities)
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise AnnotationError(f"Probabilities must sum to one, got {total}")
        if max(range(len(probabilities)), key=lambda i: probabilities[i]) != label:
            raise AnnotationError("Predicted label disagrees with probability argmax")
        probabilities = [float(p / total) for p in probabilities]
        probability_source = probability_source or "provided_distribution"
    return {"schema_version": 1, **sample, "label_id": label, "class_name": classes[label],
            "probabilities": probabilities, "probability_source": probability_source,
            "source_model": str(model), "alignment": alignment, "raw_output": raw_output}
