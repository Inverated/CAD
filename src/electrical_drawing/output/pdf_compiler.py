"""Assemble scaled drawings into a single multi-page PDF, plus SVG pages."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import schemdraw
from matplotlib.backends.backend_pdf import PdfPages

from ..configurations.constants import (
    PAGE_HEIGHT_INCHES,
    PAGE_MARGIN_INCHES,
    PAGE_WIDTH_INCHES,
)
from ..page_scaler import render_to_page, scale_to_page


def compile_pdf(
    drawings: list[schemdraw.Drawing],
    output_path: str,
    page_width_inches: float = PAGE_WIDTH_INCHES,
    page_height_inches: float = PAGE_HEIGHT_INCHES,
    margin_inches: float = PAGE_MARGIN_INCHES,
) -> str:
    """Render each drawing onto its own page of a PDF at ``output_path``.

    Drawings are scaled to the page first; already-scaled drawings are left
    alone. Returns ``output_path``.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for drawing in drawings:
            fit = scale_to_page(
                drawing,
                page_width_inches=page_width_inches,
                page_height_inches=page_height_inches,
                margin_inches=margin_inches,
            )
            figure = render_to_page(drawing, fit)
            try:
                pdf.savefig(figure)
            finally:
                plt.close(figure)

    return output_path


def close_backend_figure(drawing: schemdraw.Drawing) -> None:
    """Release the matplotlib figure schemdraw creates when saving a drawing.

    ``Drawing.save()`` builds a pyplot figure and keeps a reference to it, so
    saving many pages in one process otherwise leaks figures.
    """
    backend_figure = getattr(drawing, 'fig', None)
    matplotlib_figure = getattr(backend_figure, 'fig', None)
    if matplotlib_figure is not None:
        plt.close(matplotlib_figure)


def save_pages(
    named_drawings: Iterable[tuple[str, schemdraw.Drawing]],
    pages_dir: str,
    formats: tuple[str, ...] = ('svg',),
) -> list[str]:
    """Save each ``(name, drawing)`` individually into ``pages_dir``.

    Returns the paths written, in order.
    """
    os.makedirs(pages_dir, exist_ok=True)
    written = []
    for name, drawing in named_drawings:
        for fmt in formats:
            path = os.path.join(pages_dir, f'{name}.{fmt}')
            drawing.save(path)
            close_backend_figure(drawing)
            written.append(path)
    return written


def page_count(pdf_path: str) -> Optional[int]:
    """Count pages in a PDF by scanning for page objects.

    Avoids adding a PDF library dependency just to verify output; returns
    ``None`` if the file cannot be read.
    """
    try:
        with open(pdf_path, 'rb') as handle:
            content = handle.read()
    except OSError:
        return None

    count = content.count(b'/Type /Page')
    # Some writers emit '/Type /Pages' for the page tree node; discount those.
    count -= content.count(b'/Type /Pages')
    return count if count > 0 else content.count(b'/Type/Page') or None
