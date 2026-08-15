"""Task 6 demo: end-to-end generation for every boat.

Runs the orchestrator for rp1/rp2/rp3, reports the output tree and page counts,
and additionally renders each page to a PNG at true page size so the PDF pages
can be inspected as images.

Run:  python -m src.electrical_drawing.demos.task6_full_set
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from ..drawing_generator import generate_all
from ..output.pdf_compiler import page_count
from ..page_scaler import render_to_page
from .demo_utils import DEMO_DIR

COMPONENTS = 'constant/electrical/components.json'
BOATS = ('rp1', 'rp2', 'rp3')


def run(boat: str, preview: bool) -> None:
    output = os.path.join('artifact', f'{boat}.electrical_drawing')
    result = generate_all(
        circuit_path=f'constant/electrical/boat/{boat}/circuit_setup.json',
        components_path=COMPONENTS,
        boat_params_path=f'constant/boat/{boat}.json',
        output_path=output,
    )

    pdf_pages = page_count(result.pdf_path)
    print(f'{boat}: {result.page_count} pages -> PDF reports {pdf_pages}')
    for name, fit in zip(result.page_names, result.fits):
        status = 'fits' if fit.fits() else 'OVERFLOW'
        print(f'    {name:<24} {fit.content_width_inches:5.2f} x '
              f'{fit.content_height_inches:5.2f} in  [{status}]')
    print(f'    pdf: {result.pdf_path}')

    if not preview:
        return

    os.makedirs(DEMO_DIR, exist_ok=True)
    for name, drawing, fit in zip(result.page_names, result.drawings, result.fits):
        figure = render_to_page(drawing, fit)
        path = os.path.join(DEMO_DIR, f'task6.{name}.page.png')
        figure.savefig(path, dpi=110)
        plt.close(figure)
        print(f'    preview: {path}')


def main() -> None:
    for boat in BOATS:
        run(boat, preview=(boat == 'rp2'))
    print(f'\nDemo output directory: {DEMO_DIR}')


if __name__ == '__main__':
    main()
