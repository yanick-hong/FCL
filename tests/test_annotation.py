"""Annotation correctness checks use small synthetic identities, never pretrained models."""
import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.utils.annotation.cli import build_prompt, export_cache, import_labels
from src.utils.annotation.schema import AnnotationError, Manifest, make_record, parse_label
from src.utils.annotation.store import AnnotationStore, load_annotations


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.classes = ["cat", "dog", "car"]
        self.samples = [{"sample_id": "a", "path": "images/a.jpg", "split": "train"},
                        {"sample_id": "b", "path": "images/b.jpg", "split": "train"}]
        self.manifest = Manifest(self.samples)
        self.manifest_path = self.root / "manifest.jsonl"
        self.manifest_path.write_text("\n".join(json.dumps(s) for s in self.samples), encoding="utf-8")
        self.classes_path = self.root / "classes.json"
        self.classes_path.write_text(json.dumps(self.classes), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def import_args(self, content, **overrides):
        source = self.root / "input.txt"
        source.write_text(content, encoding="utf-8")
        values = dict(manifest=str(self.manifest_path), split="train", classes=str(self.classes_path),
                      input=str(source), output=str(self.root / "labels.jsonl"), source_model="test-model",
                      label_base=0, ordered_legacy=False, allow_partial=False, resume=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_strict_labels(self):
        self.assertEqual(parse_label('{"label_id": 2}', self.classes), 2)
        self.assertEqual(parse_label("DOG", self.classes), 1)
        self.assertEqual(parse_label("3", self.classes, 1), 2)
        for value in (True, -1, 3, 1.0, "12", "1 or 2", "dog because...", '{"label_id": 1, "label": 2}'):
            with self.subTest(value=value), self.assertRaises(AnnotationError):
                parse_label(value, self.classes)

    def test_identity_disagreement(self):
        with self.assertRaises(AnnotationError):
            self.manifest.match({"sample_id": "a", "path": "images/b.jpg"})
        with self.assertRaises(AnnotationError):
            Manifest(self.samples + [self.samples[0]])

    def test_probability_validation_and_onehot(self):
        record = make_record(self.samples[0], 2, self.classes, "exact-model")
        self.assertEqual(record["probabilities"], [0., 0., 1.])
        self.assertEqual(record["probability_source"], "one_hot_from_label")
        for probs in ([0.2, 0.1, 0.1], [0.2, 0.8, 0], [math.nan, 0, 1], [0, 1]):
            with self.subTest(probs=probs), self.assertRaises(AnnotationError):
                make_record(self.samples[0], 2, self.classes, "model", probs)

    def test_jsonl_import_aligns_by_identity(self):
        args = self.import_args('{"sample_id":"b","label_id":2}\n{"path":"images/a.jpg","label":1}\n')
        import_labels(args)
        records = load_annotations(args.output, self.manifest, self.classes)
        self.assertEqual(records["a"]["label_id"], 1)
        self.assertEqual(records["b"]["label_id"], 2)

    def test_legacy_requires_explicit_order_and_count(self):
        with self.assertRaises(AnnotationError):
            import_labels(self.import_args("0\n1\n"))
        with self.assertRaises(AnnotationError):
            import_labels(self.import_args("0\n", ordered_legacy=True))
        args = self.import_args("0\n2\n", ordered_legacy=True)
        import_labels(args)
        records = load_annotations(args.output, self.manifest, self.classes)
        self.assertEqual(records["b"]["label_id"], 2)
        self.assertEqual(records["a"]["alignment"], "explicit_legacy_order")

    def test_import_validates_before_writing(self):
        args = self.import_args("0\n9\n", ordered_legacy=True)
        with self.assertRaises(AnnotationError):
            import_labels(args)
        self.assertFalse(Path(args.output).exists())

    def test_resume_and_tail_recovery(self):
        path = self.root / "progress.jsonl"
        store = AnnotationStore(path, self.manifest, self.classes, "model", {"x": 1})
        store.append(make_record(self.samples[0], 0, self.classes, "model"))
        with path.open("ab") as handle:
            handle.write(b'{"sample_id":"b","label')
        resumed = AnnotationStore(path, self.manifest, self.classes, "model", {"x": 1}, resume=True)
        self.assertEqual(set(resumed.records), {"a"})
        self.assertTrue(Path(str(path) + ".partial-tail.bak").exists())
        resumed.append(make_record(self.samples[1], 1, self.classes, "model"))
        self.assertEqual(len(load_annotations(path, self.manifest, self.classes)), 2)
        with self.assertRaises(AnnotationError):
            AnnotationStore(path, self.manifest, self.classes, "different-model", {"x": 1}, resume=True)

    def test_duplicate_rejected(self):
        path = self.root / "progress.jsonl"
        store = AnnotationStore(path, self.manifest, self.classes, "model")
        record = make_record(self.samples[0], 0, self.classes, "model")
        store.append(record)
        with self.assertRaises(AnnotationError):
            store.append(record)

    def test_prompt_variants(self):
        for variant in ("A", "B", "C"):
            prompt = build_prompt(self.classes, variant)
            self.assertIn("0: cat", prompt)
            self.assertIn("label_id", prompt)

    def test_export_cache_identity_alignment(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional torch unavailable")
        args = self.import_args('{"sample_id":"b","label_id":2}\n{"sample_id":"a","label_id":1}\n')
        import_labels(args)
        cache_path, output = self.root / "cache.pt", self.root / "qwen.pt"
        original = torch.tensor([[1., 0, 0], [0, 1., 0]])
        torch.save({"train_feats": torch.zeros(2, 4), "train_sample_ids": ["b", "a"],
                    "clip_probs_train": original, "clip_argmax_train": torch.tensor([0, 1])}, cache_path)
        options = argparse.Namespace(manifest=str(self.manifest_path), split="train", classes=str(self.classes_path),
                                     annotations=args.output, cache=str(cache_path), output=str(output),
                                     confirm_cache_manifest_order=False, replace_clip_fields=True)
        export_cache(options)
        cache = torch.load(output, weights_only=False)
        self.assertEqual(cache["vlm_argmax_train"].tolist(), [2, 1])
        self.assertTrue(torch.equal(cache["original_clip_probs_train"], original))
        self.assertIn("one_hot_from_label", cache["annotation_metadata"]["train"]["probability_sources"])


if __name__ == "__main__":
    unittest.main()
