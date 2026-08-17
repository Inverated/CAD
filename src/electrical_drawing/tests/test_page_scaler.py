"""Task 5 tests: page scaling and PDF compilation."""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest
import schemdraw
import schemdraw.elements as elm

from ..configurations.constants import (
    BASE_FONTSIZE,
    PAGE_HEIGHT_INCHES,
    PAGE_MARGIN_INCHES,
    PAGE_WIDTH_INCHES,
)
from ..output.pdf_compiler import compile_pdf, page_count, save_pages
from ..page_scaler import (
    MAX_INCHES_PER_UNIT,
    apply_font_scale,
    compute_fit,
    measure,
    render_to_page,
    scale_to_page,
)

USABLE_WIDTH = PAGE_WIDTH_INCHES - 2 * PAGE_MARGIN_INCHES
USABLE_HEIGHT = PAGE_HEIGHT_INCHES - 2 * PAGE_MARGIN_INCHES


def chain(length: int, vertical: bool = False) -> schemdraw.Drawing:
    """A drawing of ``length`` resistors, run horizontally or vertically."""
    drawing = schemdraw.Drawing()
    drawing.config(unit=2, fontsize=BASE_FONTSIZE)
    for index in range(length):
        element = elm.Resistor().label(f'R{index + 1}')
        drawing.add(element.down() if vertical else element.right())
    return drawing


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

@pytest.mark.parametrize('length', [1, 5, 20, 60])
@pytest.mark.parametrize('vertical', [False, True])
def test_scaled_content_is_within_usable_area(length, vertical):
    fit = compute_fit(chain(length, vertical))
    assert fit.content_width_inches <= USABLE_WIDTH + 1e-6
    assert fit.content_height_inches <= USABLE_HEIGHT + 1e-6


@pytest.mark.parametrize('length', [1, 5, 20, 60])
def test_scaled_content_is_within_page_bounds(length):
    assert compute_fit(chain(length)).fits()


def test_larger_drawings_get_smaller_scales():
    small = compute_fit(chain(4))
    large = compute_fit(chain(40))
    assert large.scale < small.scale


def test_scale_is_capped_to_avoid_blowing_up_tiny_drawings():
    assert compute_fit(chain(1)).scale <= MAX_INCHES_PER_UNIT


def test_upscaling_can_be_disabled():
    fit = compute_fit(chain(1), allow_upscale=False)
    assert fit.scale <= 0.5
    assert fit.font_ratio <= 1.0


def test_custom_page_size_is_respected():
    fit = compute_fit(chain(10), page_width_inches=5, page_height_inches=4, margin_inches=0.25)
    assert fit.page_width_inches == 5
    assert fit.content_width_inches <= 5 - 2 * 0.25 + 1e-6


def test_measure_includes_padding_around_content():
    drawing = chain(3)
    bbox = drawing.get_bbox()
    width, height, centre_x, centre_y = measure(drawing)
    assert width > (bbox.xmax - bbox.xmin)
    assert height > (bbox.ymax - bbox.ymin)
    assert centre_x == pytest.approx((bbox.xmin + bbox.xmax) / 2)
    assert centre_y == pytest.approx((bbox.ymin + bbox.ymax) / 2)


# --------------------------------------------------------------------------
# Proportional text scaling
# --------------------------------------------------------------------------

def test_font_scale_multiplies_label_sizes():
    drawing = chain(3)
    before = [label.fontsize for label in drawing.elements[0]._userlabels]
    apply_font_scale(drawing, 2.0)
    after = [label.fontsize for label in drawing.elements[0]._userlabels]
    # Sizes were unset, so they inherit the drawing default before scaling.
    assert before == [None]
    assert after == [pytest.approx(BASE_FONTSIZE * 2.0)]


def test_shrinking_a_drawing_shrinks_its_text():
    drawing = chain(40)
    fit = scale_to_page(drawing)
    assert fit.font_ratio < 1.0
    sizes = [
        label.fontsize
        for element in drawing.elements
        for label in element._userlabels
    ]
    assert sizes and all(size < BASE_FONTSIZE for size in sizes)


def test_scale_to_page_is_idempotent():
    drawing = chain(20)
    first = scale_to_page(drawing)
    sizes_after_first = [
        label.fontsize for element in drawing.elements for label in element._userlabels
    ]
    second = scale_to_page(drawing)
    sizes_after_second = [
        label.fontsize for element in drawing.elements for label in element._userlabels
    ]
    assert first is second
    assert sizes_after_first == sizes_after_second


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

@pytest.mark.parametrize('length', [2, 25])
def test_rendered_figure_is_exactly_page_sized(length):
    drawing = chain(length)
    figure = render_to_page(drawing)
    try:
        width, height = figure.get_size_inches()
        assert width == pytest.approx(PAGE_WIDTH_INCHES)
        assert height == pytest.approx(PAGE_HEIGHT_INCHES)
    finally:
        plt.close(figure)


def test_rendered_axes_maps_units_to_the_computed_scale():
    drawing = chain(10)
    fit = scale_to_page(drawing)
    figure = render_to_page(drawing, fit)
    try:
        axes = figure.axes[0]
        span_units = axes.get_xlim()[1] - axes.get_xlim()[0]
        assert span_units * fit.scale == pytest.approx(PAGE_WIDTH_INCHES)
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------
# PDF compilation
# --------------------------------------------------------------------------

@pytest.mark.parametrize('pages', [1, 3, 5])
def test_pdf_has_one_page_per_drawing(tmp_path, pages):
    drawings = [chain(index + 2) for index in range(pages)]
    out = tmp_path / 'multi.pdf'
    compile_pdf(drawings, str(out))
    assert out.exists()
    assert page_count(str(out)) == pages


def test_pdf_parent_directory_is_created(tmp_path):
    out = tmp_path / 'nested' / 'deeper' / 'out.pdf'
    compile_pdf([chain(3)], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_save_pages_writes_one_file_per_drawing(tmp_path):
    named = [('page1', chain(2)), ('page2', chain(3))]
    pages_dir = tmp_path / 'pages'
    written = save_pages(named, str(pages_dir))
    assert len(written) == 2
    for path in written:
        assert path.endswith('.svg')
        assert pages_dir.joinpath(path.split('/')[-1]).exists()
