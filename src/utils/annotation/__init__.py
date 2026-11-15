"""Identity-aligned, resumable VLM annotation and legacy label import."""

from .schema import AnnotationError, Manifest, make_record, parse_label, read_classes

__all__ = ["AnnotationError", "Manifest", "make_record", "parse_label", "read_classes"]
