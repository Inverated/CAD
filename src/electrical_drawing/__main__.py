#!/usr/bin/env python3
"""Generate the multi-page electrical drawing set for one boat.

Example:
    python -m src.electrical_drawing \\
        --circuit constant/electrical/boat/rp2/circuit_setup.json \\
        --components constant/electrical/components.json \\
        --boat-params constant/boat/rp2.json \\
        --output artifact/rp2.electrical_drawing
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

# Headless-safe: this tool only ever writes files.
matplotlib.use('Agg')

from .configurations.constants import (
    PAGE_HEIGHT_INCHES,
    PAGE_MARGIN_INCHES,
    PAGE_WIDTH_INCHES,
)
from .drawing_generator import generate_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m src.electrical_drawing',
        description='Generate multi-page electrical schematics from JSON configuration',
    )
    parser.add_argument('--circuit', required=True,
                        help='Path to circuit setup JSON '
                             '(e.g. constant/electrical/boat/rp2/circuit_setup.json)')
    parser.add_argument('--components', required=True,
                        help='Path to components JSON (e.g. constant/electrical/components.json)')
    parser.add_argument('--boat-params', required=True,
                        help='Path to boat parameter JSON (e.g. constant/boat/rp2.json), used '
                             'for panel series/parallel fallbacks')
    parser.add_argument('--output', required=True,
                        help='Output directory (e.g. artifact/rp2.electrical_drawing)')
    parser.add_argument('--page-width', type=float, default=PAGE_WIDTH_INCHES,
                        help=f'Page width in inches (default: {PAGE_WIDTH_INCHES}, A4 landscape)')
    parser.add_argument('--page-height', type=float, default=PAGE_HEIGHT_INCHES,
                        help=f'Page height in inches (default: {PAGE_HEIGHT_INCHES}, A4 landscape)')
    parser.add_argument('--margin', type=float, default=PAGE_MARGIN_INCHES,
                        help=f'Page margin in inches (default: {PAGE_MARGIN_INCHES})')
    parser.add_argument('--page-format', action='append', default=None,
                        help='Individual page format, repeatable (default: svg)')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    formats = tuple(args.page_format) if args.page_format else ('svg',)

    result = generate_all(
        circuit_path=args.circuit,
        components_path=args.components,
        boat_params_path=args.boat_params,
        output_path=args.output,
        page_width_inches=args.page_width,
        page_height_inches=args.page_height,
        margin_inches=args.margin,
        page_formats=formats,
    )

    print(f'Electrical drawing: {result.boat} ({result.page_count} pages)')
    for name, fit in zip(result.page_names, result.fits):
        print(f'  {name:<28} scale {fit.scale:.3f} in/unit  '
              f'({fit.content_width_inches:.2f} x {fit.content_height_inches:.2f} in)')
    print(f'  pages -> {result.pages_dir}')
    print(f'  pdf   -> {result.pdf_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
