"""Task 4 tests: battery bus page generation."""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import pytest
import schemdraw.elements as elm

from ..components.Battery import Battery
from ..components.Load import Load
from ..configurations.config import (
    BatterySpec,
    CircuitConfig,
    LoadSpec,
    MpptGroup,
    MpptSpec,
    PanelSpec,
)
from ..drawings.battery_bus_drawing import generate_battery_bus_drawing
from .helpers import count_of_type, label_texts


def group(count: int, name: str = 'config_1') -> MpptGroup:
    return MpptGroup(
        name=name,
        count=count,
        panel=PanelSpec(choice='Renogy_455W', in_series=2, in_parallel=2),
        mppt=MpptSpec(choice='Victron_150/45'),
    )


def config(
    mppt_counts=(2,),
    battery_series=2,
    battery_parallel=1,
    load_count=1,
) -> CircuitConfig:
    return CircuitConfig(
        boat_name='rp2',
        display_name='Electric Proa',
        description='test',
        mppt_groups=[group(count, f'config_{i + 1}') for i, count in enumerate(mppt_counts)],
        loads=[
            LoadSpec(name=f'load_{i + 1}', choice=f'Motor_{i + 1}', total_power=4000)
            for i in range(load_count)
        ],
        battery=BatterySpec(
            choice='LiNMC',
            in_series=battery_series,
            in_parallel=battery_parallel,
            block_voltage=25.9,
        ),
    )


# --------------------------------------------------------------------------
# Incoming tags: two per MPPT instance
# --------------------------------------------------------------------------

@pytest.mark.parametrize('counts,expected_mppts', [
    ((1,), 1),
    ((2,), 2),
    ((3,), 3),
    ((2, 3), 5),
    ((1, 1, 1), 3),
])
def test_tag_count_is_two_per_mppt(counts, expected_mppts):
    drawing = generate_battery_bus_drawing(config(mppt_counts=counts))
    assert count_of_type(drawing, elm.Tag) == 2 * expected_mppts


def test_total_mppt_count_sums_every_config_block():
    assert config(mppt_counts=(2, 3)).total_mppt_count == 5


def test_each_mppt_has_a_positive_and_negative_tag():
    drawing = generate_battery_bus_drawing(config(mppt_counts=(3,)))
    texts = label_texts(drawing)
    for index in (1, 2, 3):
        assert any(f'MPPT{index} BATT+' in text for text in texts)
        assert any(f'MPPT{index} BATT-' in text for text in texts)


def test_incoming_tags_point_back_at_their_source():
    drawing = generate_battery_bus_drawing(config())
    tag_labels = [
        label.label
        for element in drawing.elements
        if isinstance(element, elm.Tag)
        for label in element._userlabels
    ]
    assert tag_labels
    assert all(label.startswith('← ') for label in tag_labels)


# --------------------------------------------------------------------------
# Battery bank
# --------------------------------------------------------------------------

@pytest.mark.parametrize('series,parallel', [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2)])
def test_battery_count_matches_config(series, parallel):
    drawing = generate_battery_bus_drawing(
        config(battery_series=series, battery_parallel=parallel)
    )
    assert count_of_type(drawing, Battery) == series * parallel


def test_battery_summary_in_title():
    drawing = generate_battery_bus_drawing(config(battery_series=2, battery_parallel=2))
    joined = '\n'.join(label_texts(drawing))
    assert '2s2p' in joined
    assert 'LiNMC' in joined


# --------------------------------------------------------------------------
# Loads
# --------------------------------------------------------------------------

@pytest.mark.parametrize('load_count', [0, 1, 2, 3])
def test_load_count_matches_config(load_count):
    drawing = generate_battery_bus_drawing(config(load_count=load_count))
    assert count_of_type(drawing, Load) == load_count


def test_loads_are_numbered_sequentially():
    drawing = generate_battery_bus_drawing(config(load_count=3))
    texts = label_texts(drawing)
    for number in (1, 2, 3):
        assert any(text == f'M{number}' for text in texts)


# --------------------------------------------------------------------------
# Bus bars
# --------------------------------------------------------------------------

def test_both_bus_bars_are_labelled():
    texts = label_texts(generate_battery_bus_drawing(config()))
    assert 'B+' in texts
    assert 'B-' in texts


def test_bus_bars_span_every_tap():
    """Both bus bars are horizontal, share their extent, and are vertically apart."""
    drawing = generate_battery_bus_drawing(config(mppt_counts=(3,)))
    bars = [
        element
        for element in drawing.elements
        if isinstance(element, elm.Line)
        and any(label.label in ('B+', 'B-') for label in element._userlabels)
    ]
    assert len(bars) == 2
    positive, negative = bars
    assert positive.start.y == pytest.approx(positive.end.y)
    assert negative.start.y == pytest.approx(negative.end.y)
    assert positive.start.y > negative.start.y
    assert positive.start.x == pytest.approx(negative.start.x)
    assert positive.end.x == pytest.approx(negative.end.x)


def test_bus_gap_widens_for_a_taller_battery_stack():
    short = generate_battery_bus_drawing(config(battery_series=1)).get_bbox()
    tall = generate_battery_bus_drawing(config(battery_series=6)).get_bbox()
    assert (tall.ymax - tall.ymin) > (short.ymax - short.ymin)


def test_page_renders_to_file(tmp_path):
    out = tmp_path / 'bus.svg'
    generate_battery_bus_drawing(config()).save(str(out))
    assert out.exists() and out.stat().st_size > 0
