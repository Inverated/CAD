"""Fit a drawing onto a fixed page size.

Matplotlib sizes geometry in data units but text and line widths in points, so
shrinking a drawing to fit a page requires scaling all three together or the
proportions drift — text ends up enormous on a large schematic and hairline-thin
lines on a small one.

:func:`scale_to_page` therefore does three things:

1. computes the inches-per-unit needed for the content to fit inside the page
   margins,
2. multiplies every font size and the line width by the same ratio,
3. records the result on the drawing so the work is never repeated.

:func:`render_to_page` then draws onto an axes that fills a true page-sized
figure, with data limits chosen so the mapping from drawing units to inches is
exactly the computed scale. Content ends up centred on a real A4 page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import schemdraw
from schemdraw.segments import SegmentText

from .configurations.constants import (
    BASE_FONTSIZE,
    BASE_INCHES_PER_UNIT,
    PAGE_HEIGHT_INCHES,
    PAGE_MARGIN_INCHES,
    PAGE_WIDTH_INCHES,
)

# Default schemdraw line width, in points.
BASE_LINEWIDTH = 2

# Never let lines or text disappear entirely on a heavily shrunk page.
MIN_LINEWIDTH = 0.4
MIN_FONTSIZE = 2.0

# Cap on enlargement so a tiny schematic is not blown up to fill a page.
MAX_INCHES_PER_UNIT = 1.0

# Padding added around the content, in drawing units.
CONTENT_MARGIN_UNITS = 0.3

_FIT_ATTR = '_electrical_drawing_page_fit'


@dataclass(frozen=True)
class PageFit:
    """The result of fitting a drawing to a page."""

    scale: float
    font_ratio: float
    page_width_inches: float
    page_height_inches: float
    content_width_units: float
    content_height_units: float
    centre_x: float
    centre_y: float

    @property
    def content_width_inches(self) -> float:
        return self.content_width_units * self.scale

    @property
    def content_height_inches(self) -> float:
        return self.content_height_units * self.scale

    def fits(self, tolerance: float = 1e-6) -> bool:
        """True when the scaled content lies within the page bounds."""
        return (
            self.content_width_inches <= self.page_width_inches + tolerance
            and self.content_height_inches <= self.page_height_inches + tolerance
        )


def measure(drawing: schemdraw.Drawing, margin_units: float = CONTENT_MARGIN_UNITS):
    """Return ``(width_units, height_units, centre_x, centre_y)`` for the content."""
    bbox = drawing.get_bbox()
    width = (bbox.xmax - bbox.xmin) + 2 * margin_units
    height = (bbox.ymax - bbox.ymin) + 2 * margin_units
    centre_x = (bbox.xmin + bbox.xmax) / 2
    centre_y = (bbox.ymin + bbox.ymax) / 2
    return max(width, 1e-6), max(height, 1e-6), centre_x, centre_y


def compute_fit(
    drawing: schemdraw.Drawing,
    page_width_inches: float = PAGE_WIDTH_INCHES,
    page_height_inches: float = PAGE_HEIGHT_INCHES,
    margin_inches: float = PAGE_MARGIN_INCHES,
    allow_upscale: bool = True,
) -> PageFit:
    """Work out the scale that fits ``drawing`` inside the page margins."""
    width_units, height_units, centre_x, centre_y = measure(drawing)

    usable_width = max(page_width_inches - 2 * margin_inches, 1e-6)
    usable_height = max(page_height_inches - 2 * margin_inches, 1e-6)

    scale = min(usable_width / width_units, usable_height / height_units)
    if not allow_upscale:
        scale = min(scale, BASE_INCHES_PER_UNIT)
    scale = min(scale, MAX_INCHES_PER_UNIT)

    return PageFit(
        scale=scale,
        font_ratio=scale / BASE_INCHES_PER_UNIT,
        page_width_inches=page_width_inches,
        page_height_inches=page_height_inches,
        content_width_units=width_units,
        content_height_units=height_units,
        centre_x=centre_x,
        centre_y=centre_y,
    )


def apply_font_scale(drawing: schemdraw.Drawing, ratio: float) -> None:
    """Multiply every font size on the drawing by ``ratio``.

    Covers both labels added with ``.label()`` and text baked into elements as
    segments (such as the MPPT pin names). Sizes left unset inherit the
    drawing's configured default before scaling.
    """
    default = drawing.dwgparams.get('fontsize', BASE_FONTSIZE)

    for element in drawing.elements:
        for label in getattr(element, '_userlabels', []):
            label.fontsize = max(MIN_FONTSIZE, (label.fontsize or default) * ratio)
        for segment in element.segments:
            if isinstance(segment, SegmentText):
                segment.fontsize = max(MIN_FONTSIZE, (segment.fontsize or default) * ratio)


def scale_to_page(
    drawing: schemdraw.Drawing,
    page_width_inches: float = PAGE_WIDTH_INCHES,
    page_height_inches: float = PAGE_HEIGHT_INCHES,
    margin_inches: float = PAGE_MARGIN_INCHES,
    allow_upscale: bool = True,
) -> PageFit:
    """Scale ``drawing`` so it fits on one page, and return the fit.

    Idempotent: calling this again returns the fit computed the first time
    without rescaling the fonts a second time.
    """
    existing = getattr(drawing, _FIT_ATTR, None)
    if existing is not None:
        return existing

    fit = compute_fit(
        drawing,
        page_width_inches=page_width_inches,
        page_height_inches=page_height_inches,
        margin_inches=margin_inches,
        allow_upscale=allow_upscale,
    )

    apply_font_scale(drawing, fit.font_ratio)
    drawing.config(
        inches_per_unit=fit.scale,
        fontsize=max(MIN_FONTSIZE, BASE_FONTSIZE * fit.font_ratio),
        lw=max(MIN_LINEWIDTH, BASE_LINEWIDTH * fit.font_ratio),
    )

    setattr(drawing, _FIT_ATTR, fit)
    return fit


def render_to_page(drawing: schemdraw.Drawing, fit: Optional[PageFit] = None):
    """Draw ``drawing`` centred on a page-sized matplotlib figure.

    The axes fills the figure and its data limits are set so that one drawing
    unit maps to exactly ``fit.scale`` inches, which keeps the scaled text and
    line widths proportional to the geometry.
    """
    if fit is None:
        fit = scale_to_page(drawing)

    figure = plt.figure(figsize=(fit.page_width_inches, fit.page_height_inches))
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_aspect('equal')
    axes.axis('off')

    half_width_units = fit.page_width_inches / (2 * fit.scale)
    half_height_units = fit.page_height_inches / (2 * fit.scale)
    axes.set_xlim(fit.centre_x - half_width_units, fit.centre_x + half_width_units)
    axes.set_ylim(fit.centre_y - half_height_units, fit.centre_y + half_height_units)

    drawing.draw(canvas=axes, show=False)
    return figure
