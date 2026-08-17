"""Shared helpers for the electrical drawing demos."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use('Agg')

import schemdraw

DEMO_DIR = os.path.join('artifact', 'electrical_drawing_demos')


def save(drawing: schemdraw.Drawing, name: str, formats: tuple[str, ...] = ('svg', 'png')) -> str:
    """Save ``drawing`` into the demo directory in each requested format.

    Returns the path of the first format written.
    """
    os.makedirs(DEMO_DIR, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(DEMO_DIR, f'{name}.{fmt}')
        drawing.save(path)
        paths.append(path)
    return paths[0]
