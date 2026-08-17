"""Task 3 tests: MPPT + panel page generation."""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import pytest
import schemdraw.elements as elm

from ..components.MPPT import MPPT
from ..configurations.config import MpptSpec, PanelSpec
from ..drawings.mppt_panel_drawing import generate_mppt_panel_drawing
from .helpers import all_texts, count_of_type, label_texts

MPPT_MODEL = MpptSpec(choice='Victron_250/100')


def panel(series: int, parallel: int) -> PanelSpec:
    return PanelSpec(
        choice='Renogy_455W',
        in_series=series,
        in_parallel=parallel,
        power_watts=455,
        voltage=41.5,
    )


def page(series: int = 4, parallel: int = 2, index: int = 1):
    return generate_mppt_panel_drawing(
        mppt_index=index,
        panel=panel(series, parallel),
        mppt=MPPT_MODEL,
    )


# --------------------------------------------------------------------------
# Element counts
# --------------------------------------------------------------------------

@pytest.mark.parametrize('series,parallel', [(1, 1), (1, 2), (2, 2), (4, 2), (3, 1)])
def test_solar_panel_count_matches_config(series, parallel):
    drawing = page(series, parallel)
    assert count_of_type(drawing, elm.Solar) == series * parallel


@pytest.mark.parametrize('series,parallel', [(1, 2), (4, 2)])
def test_exactly_one_mppt(series, parallel):
    assert count_of_type(page(series, parallel), MPPT) == 1


@pytest.mark.parametrize('series,parallel', [(1, 2), (4, 2)])
def test_exactly_two_tags(series, parallel):
    assert count_of_type(page(series, parallel), elm.Tag) == 2


# --------------------------------------------------------------------------
# Tag content
# --------------------------------------------------------------------------

def test_tags_reference_both_battery_bus_rails():
    texts = label_texts(page())
    assert any('Battery Bus B+' in text for text in texts)
    assert any('Battery Bus B-' in text for text in texts)


def test_tags_point_outwards():
    """Outgoing tags carry the forward arrow."""
    tag_labels = [
        label.label
        for element in page().elements
        if isinstance(element, elm.Tag)
        for label in element._userlabels
    ]
    assert len(tag_labels) == 2
    assert all(label.startswith('→ ') for label in tag_labels)


@pytest.mark.parametrize('index', [1, 2, 3])
def test_page_identifies_its_mppt_index(index):
    """The MPPT index appears on the page (body text and title)."""
    texts = all_texts(page(index=index))
    assert any(f'MPPT{index}' in text for text in texts)


# --------------------------------------------------------------------------
# Labels and title
# --------------------------------------------------------------------------

def test_panels_are_numbered_sequentially():
    texts = all_texts(page(4, 2))
    for number in range(1, 9):
        assert any(f'PV{number}' == text for text in texts)


def test_title_includes_panel_and_mppt_spec():
    texts = label_texts(page(4, 2))
    joined = '\n'.join(texts)
    assert '4s2p' in joined
    assert 'Renogy_455W' in joined
    assert 'Victron_250/100' in joined


def test_boat_label_appears_when_given():
    drawing = generate_mppt_panel_drawing(1, panel(2, 2), MPPT_MODEL, boat_label='rp2')
    assert any('rp2' in text for text in label_texts(drawing))


def test_mppt_pins_align_with_panel_array_terminals():
    """PV+ / PV- sit exactly one terminating distance apart, matching the array."""
    drawing = page(4, 2)
    converter = next(e for e in drawing.elements if isinstance(e, MPPT))
    positive = converter.absanchors['PV+']
    negative = converter.absanchors['PV-']
    assert positive.x == pytest.approx(negative.x)
    assert positive.y - negative.y == pytest.approx(2)


def test_different_configs_produce_different_page_sizes():
    small = page(1, 2).get_bbox()
    large = page(4, 2).get_bbox()
    assert (large.ymax - large.ymin) > (small.ymax - small.ymin)


def test_page_renders_to_file(tmp_path):
    out = tmp_path / 'mppt.svg'
    page(4, 2).save(str(out))
    assert out.exists() and out.stat().st_size > 0
