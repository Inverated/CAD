"""Task 5 demo: page scaling and multi-page PDF compilation.

Builds three drawings of deliberately different sizes, fits each to A4
landscape, compiles them into a 3-page PDF, and reports the measured page and
content dimensions so the fit can be checked numerically. Also renders page 1
to PNG so the letterboxing is visible.

Run:  python -m src.electrical_drawing.demos.task5_page_pdf
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm

from ..configurations.constants import (
    PAGE_HEIGHT_INCHES,
    PAGE_MARGIN_INCHES,
    PAGE_WIDTH_INCHES,
)
from ..output.pdf_compiler import compile_pdf, page_count
from ..page_scaler import compute_fit, render_to_page, scale_to_page
from .demo_utils import DEMO_DIR


def ladder(rungs: int) -> schemdraw.Drawing:
    """A drawing whose size grows with ``rungs``, for exercising the scaler."""
    drawing = schemdraw.Drawing()
    drawing.config(unit=2, fontsize=10)
    for index in range(rungs):
        drawing.add(elm.Resistor().right().label(f'R{index + 1}'))
        if index % 2:
            drawing.add(elm.Capacitor().down().label(f'C{index + 1}'))
        else:
            drawing.add(elm.Inductor().up().label(f'L{index + 1}'))
    return drawing


def main() -> None:
    os.makedirs(DEMO_DIR, exist_ok=True)

    drawings = [ladder(rungs) for rungs in (2, 8, 30)]

    print(f'Target page: {PAGE_WIDTH_INCHES} x {PAGE_HEIGHT_INCHES} in '
          f'(A4 landscape), margin {PAGE_MARGIN_INCHES} in')
    usable = (PAGE_WIDTH_INCHES - 2 * PAGE_MARGIN_INCHES,
              PAGE_HEIGHT_INCHES - 2 * PAGE_MARGIN_INCHES)
    print(f'Usable area: {usable[0]:.2f} x {usable[1]:.2f} in\n')

    for number, drawing in enumerate(drawings, start=1):
        natural = compute_fit(drawing)
        fit = scale_to_page(drawing)
        print(f'Page {number}: content {fit.content_width_units:.1f} x '
              f'{fit.content_height_units:.1f} units')
        print(f'  scale {fit.scale:.4f} in/unit  (font ratio {fit.font_ratio:.3f})')
        print(f'  scaled to {fit.content_width_inches:.2f} x '
              f'{fit.content_height_inches:.2f} in  -> fits: {fit.fits()}')
        assert natural.scale == fit.scale

    pdf_path = os.path.join(DEMO_DIR, 'task5.three_pages.pdf')
    compile_pdf(drawings, pdf_path)
    print(f'\nCompiled {pdf_path}  (pages: {page_count(pdf_path)})')

    # Render one page to PNG so the A4 letterboxing is visible.
    figure = render_to_page(drawings[1])
    png_path = os.path.join(DEMO_DIR, 'task5.page2_on_a4.png')
    figure.savefig(png_path, dpi=100)
    plt.close(figure)
    print(f'Rendered {png_path} at {PAGE_WIDTH_INCHES} x {PAGE_HEIGHT_INCHES} in')
    print(f'\nDemo output directory: {DEMO_DIR}')


if __name__ == '__main__':
    main()
