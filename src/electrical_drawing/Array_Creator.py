"""Series/parallel component array drawing.

Draws an arbitrary ``series`` x ``parallel`` array of a two-terminal schemdraw
element and returns structured metadata describing the array's two output
terminals, so callers can wire the array into a larger schematic without having
to reverse-engineer the internal geometry.

Layout convention
-----------------
Components run *downwards* within a string, and successive parallel strings are
placed to the side (right when ``isRight`` is True, left otherwise). The two
output terminals are taken from the last string's rails and are pulled to a
fixed vertical separation of ``terminateDist`` so that downstream elements
(e.g. an MPPT with a known pin gap) line up exactly.

The *positive* terminal is the top rail, the *negative* terminal is the bottom
rail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Type

import schemdraw
import schemdraw.elements as elm
from schemdraw.util import Point

from .configurations.constants import COMPONENT_DISTANCE, TERMINATING_DISTANCE


@dataclass(frozen=True)
class TerminalPair:
    """The two connection points exposed by a drawn component array."""

    positive: Point
    negative: Point
    positive_label: str
    negative_label: str

    @property
    def gap(self) -> float:
        """Vertical separation between the positive and negative terminals."""
        return abs(self.positive.y - self.negative.y)


def component_labels(
    label_prefix: str,
    series: int,
    parallel: int,
    element_label: Optional[str] = None,
    start_index: int = 1,
) -> list[str]:
    """Return the sequential designators used for each component in the array.

    Numbering runs string-by-string, matching the draw order used by
    :func:`draw_array`, and begins at ``start_index`` so that several arrays can
    share one continuous numbering sequence. ``element_label`` is appended on a
    second line when given (e.g. ``"B1\\n12V"``).
    """
    series = max(1, series)
    parallel = max(1, parallel)

    labels = []
    for number in range(start_index, start_index + series * parallel):
        text = f'{label_prefix}{number}'
        if element_label:
            text = f'{text}\n{element_label}'
        labels.append(text)
    return labels


def draw_array(
    drawing: schemdraw.Drawing,
    element: Type[elm.Element],
    series: int,
    parallel: int,
    terminateDist: float = TERMINATING_DISTANCE,
    isRight: bool = True,
    label_prefix: str = 'B',
    element_label: Optional[str] = None,
    element_kwargs: Optional[dict] = None,
    spacing: Optional[float] = None,
    start_index: int = 1,
) -> tuple[schemdraw.Drawing, TerminalPair]:
    """Draw a ``series`` x ``parallel`` array of ``element`` onto ``drawing``.

    Parameters
    ----------
    drawing:
        Drawing to add the array to. The array starts at the drawing's current
        position.
    element:
        Two-terminal element *class* (not instance) to repeat, e.g.
        ``elm.Battery`` or the project's ``Battery`` / ``Load`` elements.
    series, parallel:
        Array dimensions. Values below 1 are clamped to 1.
    terminateDist:
        Required vertical separation of the returned terminals.
    isRight:
        Direction the rails and terminals extend.
    label_prefix:
        Designator prefix for the per-component labels (``"B"`` -> ``B1``,
        ``B2``, ...). Also used for the terminal labels (``B+`` / ``B-``).
    element_label:
        Optional second label line applied to every component (e.g. ``"12V"``).
    element_kwargs:
        Extra keyword arguments forwarded to each ``element(...)`` construction.
    spacing:
        Horizontal distance between parallel strings. Defaults to
        ``COMPONENT_DISTANCE``; widen it for elements with broad labels so that
        adjacent strings do not overlap.
    start_index:
        First designator number, letting consecutive arrays continue one
        numbering sequence.

    Returns
    -------
    tuple of (drawing, TerminalPair)
    """
    element_kwargs = element_kwargs or {}
    if spacing is None:
        spacing = COMPONENT_DISTANCE

    series = max(1, series)
    parallel = max(1, parallel)

    labels = component_labels(label_prefix, series, parallel, element_label, start_index)

    upperWire: list[elm.Element] = []
    lowerWire: list[elm.Element] = []
    label_index = 0

    for i in range(parallel):
        component_row: list[elm.Element] = []
        for _ in range(series):
            component = element(**element_kwargs).down().label(labels[label_index])
            label_index += 1
            component_row.append(component)

        fst = component_row[0]
        lst = component_row[-1]

        if i == 0:
            drawing.add(fst)
        else:
            drawing.add(fst.at(upperWire[-1].end))

        for index in range(1, len(component_row)):
            drawing.add(
                elm.Line()
                .down()
                .at(component_row[index - 1].end)
                .length(max(0, COMPONENT_DISTANCE - 2))
            )
            drawing.add(component_row[index])

        upper = (
            elm.Line().right().at(fst.start).length(spacing)
            if isRight
            else elm.Line().left().at(fst.start).length(spacing)
        )
        lower = (
            elm.Line().right().at(lst.end).length(spacing)
            if isRight
            else elm.Line().left().at(lst.end).length(spacing)
        )
        drawing.add(upper)
        drawing.add(lower)
        upperWire.append(upper)
        lowerWire.append(lower)

    gap = math.dist(lowerWire[-1].end, upperWire[-1].end)

    if math.isclose(gap, terminateDist, abs_tol=1e-9):
        # Already the requested separation: use the rails directly, which avoids
        # an unnecessary dogleg at the array output.
        top_terminal = upperWire[-1]
        bottom_terminal = lowerWire[-1]

    elif gap > terminateDist:
        topExt = elm.Line().down().at(upperWire[-1].end).length(gap / 2 - terminateDist / 2)
        drawing.add(topExt)
        top_terminal = (
            elm.Line().right().at(topExt.end) if isRight else elm.Line().left().at(topExt.end)
        )
        drawing.add(top_terminal)

        bottExt = elm.Line().up().at(lowerWire[-1].end).length(gap / 2 - terminateDist / 2)
        drawing.add(bottExt)
        bottom_terminal = (
            elm.Line().right().at(bottExt.end) if isRight else elm.Line().left().at(bottExt.end)
        )
        drawing.add(bottom_terminal)

    else:
        topExt = elm.Line().up().at(upperWire[-1].end).length(terminateDist / 2 - gap / 2)
        drawing.add(topExt)
        top_terminal = (
            elm.Line().right().at(topExt.end) if isRight else elm.Line().left().at(topExt.end)
        )
        drawing.add(top_terminal)

        bottExt = elm.Line().down().at(lowerWire[-1].end).length(terminateDist / 2 - gap / 2)
        drawing.add(bottExt)
        bottom_terminal = (
            elm.Line().right().at(bottExt.end) if isRight else elm.Line().left().at(bottExt.end)
        )
        drawing.add(bottom_terminal)

    terminals = TerminalPair(
        positive=Point(top_terminal.end),
        negative=Point(bottom_terminal.end),
        positive_label=f'{label_prefix}+',
        negative_label=f'{label_prefix}-',
    )

    return drawing, terminals
