"""Shared introspection helpers for the drawing tests."""

from __future__ import annotations

import schemdraw
from schemdraw.segments import SegmentText


def elements_of_type(drawing: schemdraw.Drawing, cls) -> list:
    """All elements on ``drawing`` that are instances of ``cls``."""
    return [element for element in drawing.elements if isinstance(element, cls)]


def count_of_type(drawing: schemdraw.Drawing, cls) -> int:
    return len(elements_of_type(drawing, cls))


def label_texts(drawing: schemdraw.Drawing) -> list[str]:
    """Every label string applied via ``.label()`` anywhere on the drawing."""
    texts = []
    for element in drawing.elements:
        for label in getattr(element, '_userlabels', []):
            if isinstance(label.label, str):
                texts.append(label.label)
            else:
                texts.extend(str(part) for part in label.label)
    return texts


def segment_texts(drawing: schemdraw.Drawing) -> list[str]:
    """Every string baked into an element as a text segment (e.g. MPPT pin names)."""
    texts = []
    for element in drawing.elements:
        for segment in element.segments:
            if isinstance(segment, SegmentText):
                texts.append(segment.text)
    return texts


def all_texts(drawing: schemdraw.Drawing) -> list[str]:
    return label_texts(drawing) + segment_texts(drawing)
