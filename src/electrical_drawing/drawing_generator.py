"""Orchestrator: configuration in, multi-page drawing set out.

Produces one page per MPPT instance plus the shared battery bus page, scales
every page to fit, then writes both the individual SVGs and the combined PDF:

    {output}/
    ├── {boat}.electrical_drawing.pdf
    └── pages/
        ├── {boat}.mppt1_panel.svg
        ├── {boat}.mppt2_panel.svg
        └── {boat}.battery_bus.svg
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import schemdraw

from .configurations.config import CircuitConfig, load_circuit
from .configurations.constants import (
    PAGE_HEIGHT_INCHES,
    PAGE_MARGIN_INCHES,
    PAGE_WIDTH_INCHES,
)
from .drawings.battery_bus_drawing import generate_battery_bus_drawing
from .drawings.mppt_panel_drawing import generate_mppt_panel_drawing
from .output.pdf_compiler import compile_pdf, save_pages
from .page_scaler import PageFit, scale_to_page

PAGES_SUBDIR = 'pages'
PDF_SUFFIX = 'electrical_drawing.pdf'


@dataclass
class GeneratedDrawings:
    """Everything produced by a run."""

    boat: str
    pdf_path: str
    pages_dir: str
    page_names: list[str] = field(default_factory=list)
    page_paths: list[str] = field(default_factory=list)
    fits: list[PageFit] = field(default_factory=list)
    drawings: list[schemdraw.Drawing] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.page_names)


def infer_boat_name(
    config: CircuitConfig,
    circuit_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Work out the boat name from the config, the output path, or the input path."""
    if config.boat_name:
        return str(config.boat_name)

    if output_path:
        # e.g. 'artifact/rp2.electrical_drawing' -> 'rp2'
        basename = os.path.basename(os.path.normpath(output_path))
        if basename:
            return basename.split('.')[0]

    if circuit_path:
        # e.g. 'constant/electrical/boat/rp2/circuit_setup.json' -> 'rp2'
        parent = os.path.basename(os.path.dirname(os.path.normpath(circuit_path)))
        if parent:
            return parent

    return 'boat'


def build_drawings(
    config: CircuitConfig,
    boat: str,
) -> list[tuple[str, schemdraw.Drawing]]:
    """Build every page for ``config`` as ``(page name, drawing)`` pairs.

    One page per MPPT instance — a ``config_N`` block with ``count: 3`` yields
    three pages — followed by the single shared battery bus page.
    """
    pages: list[tuple[str, schemdraw.Drawing]] = []

    for index, group in config.mppt_instances():
        drawing = generate_mppt_panel_drawing(
            mppt_index=index,
            panel=group.panel,
            mppt=group.mppt,
            boat_label=boat,
        )
        pages.append((f'{boat}.mppt{index}_panel', drawing))

    pages.append((f'{boat}.battery_bus', generate_battery_bus_drawing(config, boat_label=boat)))

    return pages


def generate_all(
    circuit_path: str,
    components_path: str,
    boat_params_path: Optional[str],
    output_path: str,
    page_width_inches: float = PAGE_WIDTH_INCHES,
    page_height_inches: float = PAGE_HEIGHT_INCHES,
    margin_inches: float = PAGE_MARGIN_INCHES,
    page_formats: tuple[str, ...] = ('svg',),
) -> GeneratedDrawings:
    """Load configuration, build every page, and write the SVGs and PDF."""
    config = load_circuit(circuit_path, components_path, boat_params_path)
    boat = infer_boat_name(config, circuit_path, output_path)

    pages = build_drawings(config, boat)

    fits = [
        scale_to_page(
            drawing,
            page_width_inches=page_width_inches,
            page_height_inches=page_height_inches,
            margin_inches=margin_inches,
        )
        for _, drawing in pages
    ]

    os.makedirs(output_path, exist_ok=True)
    pages_dir = os.path.join(output_path, PAGES_SUBDIR)
    page_paths = save_pages(pages, pages_dir, formats=page_formats)

    pdf_path = os.path.join(output_path, f'{boat}.{PDF_SUFFIX}')
    compile_pdf(
        [drawing for _, drawing in pages],
        pdf_path,
        page_width_inches=page_width_inches,
        page_height_inches=page_height_inches,
        margin_inches=margin_inches,
    )

    return GeneratedDrawings(
        boat=boat,
        pdf_path=pdf_path,
        pages_dir=pages_dir,
        page_names=[name for name, _ in pages],
        page_paths=page_paths,
        fits=fits,
        drawings=[drawing for _, drawing in pages],
    )
