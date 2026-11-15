"""Append-only annotation progress with identity and provenance checks."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .schema import AnnotationError, fingerprint, make_record


class AnnotationStore:
    def __init__(self, path, manifest, classes, source_model, settings=None, resume=False):
        self.path = Path(path)
        self.manifest, self.classes, self.source_model = manifest, classes, source_model
        self.records = {}
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        self.meta = {"schema_version": 1, "manifest_sha256": manifest.sha256,
                     "classes_sha256": fingerprint(classes), "class_names": classes,
                     "source_model": source_model, "settings": settings or {}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.meta_path.exists():
            if not resume:
                raise AnnotationError(f"Output already exists; use --resume: {self.path}")
            if not self.meta_path.exists() or json.loads(self.meta_path.read_text(encoding="utf-8")) != self.meta:
                raise AnnotationError("Resume provenance mismatch (manifest/classes/model/settings)")
            if not self.path.exists():
                self.path.touch()
            self._recover_tail()
            self.records = load_annotations(self.path, manifest, classes, source_model)
        else:
            self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.path.touch()

    def _recover_tail(self):
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        tail_start = data.rfind(b"\n") + 1
        try:
            json.loads(data[tail_start:])
        except (json.JSONDecodeError, UnicodeDecodeError):
            backup = self.path.with_suffix(self.path.suffix + ".partial-tail.bak")
            if backup.exists():
                raise AnnotationError(f"Unresolved partial-tail backup exists: {backup}")
            backup.write_bytes(data[tail_start:])
            self.path.write_bytes(data[:tail_start])
        else:
            with self.path.open("ab") as handle:
                handle.write(b"\n")

    def append(self, record):
        sample = self.manifest.match(record)
        key = sample["sample_id"]
        if key in self.records:
            raise AnnotationError(f"Duplicate annotation: {key}")
        checked = make_record(sample, record["label_id"], self.classes, self.source_model,
                              record.get("probabilities"), record.get("raw_output"),
                              record.get("alignment", "identity"), record.get("probability_source"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(checked, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[key] = checked


def load_annotations(path, manifest, classes, source_model=None):
    result = {}
    path = Path(path)
    if not path.exists():
        return result
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("manifest_sha256") != manifest.sha256 or meta.get("classes_sha256") != fingerprint(classes):
            raise AnnotationError("Annotation metadata belongs to a different manifest or class order")
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                sample = manifest.match(raw)
                if source_model is not None and raw.get("source_model") != source_model:
                    raise AnnotationError("Source model mismatch")
                checked = make_record(sample, raw["label_id"], classes, raw.get("source_model", "unknown"),
                                      raw.get("probabilities"), raw.get("raw_output"),
                                      raw.get("alignment", "identity"), raw.get("probability_source"))
                if "class_name" in raw and raw["class_name"] != checked["class_name"]:
                    raise AnnotationError("Saved class_name disagrees with the requested class order")
                key = checked["sample_id"]
                if key in result:
                    raise AnnotationError(f"Duplicate annotation identity: {key}")
                result[key] = checked
            except (ValueError, KeyError, TypeError) as exc:
                raise AnnotationError(f"Invalid annotation line {line_number}: {exc}") from exc
    return result
