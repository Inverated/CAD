"""Task 1 tests: Array_Creator structured terminal output."""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import pytest
import schemdraw
import schemdraw.elements as elm

from ..Array_Creator import TerminalPair, component_labels, draw_array
from ..configurations.constants import TERMINATING_DISTANCE


def build(series: int, parallel: int, terminate: float = TERMINATING_DISTANCE, **kwargs):
    drawing = schemdraw.Drawing()
    drawing.config(unit=2)
    return draw_array(
        drawing=drawing,
        element=elm.Battery,
        series=series,
        parallel=parallel,
        terminateDist=terminate,
        **kwargs,
    )


@pytest.mark.parametrize('series,parallel', [(1, 1), (2, 2), (3, 1), (4, 2)])
def test_returns_terminal_pair(series, parallel):
    _, terminals = build(series, parallel)
    assert isinstance(terminals, TerminalPair)
    assert terminals.positive is not None
    assert terminals.negative is not None


@pytest.mark.parametrize('series,parallel', [(1, 1), (2, 2), (3, 1), (4, 2)])
def test_positive_is_above_negative(series, parallel):
    """The positive terminal is the top rail, so its y is always greater."""
    _, terminals = build(series, parallel)
    assert terminals.positive.y > terminals.negative.y


@pytest.mark.parametrize('series,parallel', [(1, 1), (2, 2), (3, 1), (4, 2)])
def test_terminals_are_vertically_aligned(series, parallel):
    _, terminals = build(series, parallel)
    assert terminals.positive.x == pytest.approx(terminals.negative.x)


@pytest.mark.parametrize('series,parallel', [(1, 1), (2, 2), (3, 1), (4, 2)])
@pytest.mark.parametrize('terminate', [1.0, 2.0, 3.5])
def test_terminal_gap_equals_terminate_dist(series, parallel, terminate):
    _, terminals = build(series, parallel, terminate=terminate)
    assert terminals.gap == pytest.approx(terminate)


def test_terminal_labels_follow_prefix():
    _, terminals = build(2, 1, label_prefix='BATT')
    assert terminals.positive_label == 'BATT+'
    assert terminals.negative_label == 'BATT-'


def test_label_numbering_is_sequential():
    assert component_labels('B', 3, 1) == ['B1', 'B2', 'B3']
    assert component_labels('B', 2, 2) == ['B1', 'B2', 'B3', 'B4']
    assert component_labels('M', 1, 3) == ['M1', 'M2', 'M3']


def test_label_numbering_covers_every_component():
    labels = component_labels('B', 4, 2)
    assert len(labels) == 8
    assert labels[-1] == 'B8'


def test_element_label_added_as_second_line():
    assert component_labels('B', 1, 1, '12V') == ['B1\n12V']


def test_dimensions_are_clamped_to_at_least_one():
    _, terminals = build(0, 0)
    assert terminals.gap == pytest.approx(TERMINATING_DISTANCE)
    assert component_labels('B', 0, 0) == ['B1']


def test_left_facing_array_terminals_extend_left():
    _, right = build(2, 1, isRight=True)
    _, left = build(2, 1, isRight=False)
    assert left.positive.x < right.positive.x
