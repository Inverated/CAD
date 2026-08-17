"""Battery bus page: every MPPT output, the battery bank and the loads.

This is the single shared page that all MPPT pages point at. Two horizontal bus
bars run across it — positive on top, negative below — and everything taps onto
them:

* incoming tags on the left, one pair per MPPT instance, each dropping onto its
  bus bar through a vertical lead with a junction dot
* the battery bank, wired between the two bars
* each load, wired between the two bars in parallel with the battery

Banks are drawn as vertical strings that tap both bars directly, rather than
through :func:`~src.electrical_drawing.Array_Creator.draw_array`. On this page the
bus bars are themselves the paralleling node, so each string can run straight
down from ``B+`` to ``B-`` with no horizontal rails. That keeps every lead
straight, gives each string its own junction dot, and keeps the page narrow. The
rail-based ``draw_array`` layout is still what the MPPT pages use, where the two
terminals genuinely do have to emerge on one side at a fixed pin gap.

Bus bars are drawn last so they can be sized to span every tap point.
"""

from __future__ import annotations

from typing import Optional, Type

import schemdraw
import schemdraw.elements as elm
from schemdraw.util import Point

from ..components.Battery import Battery
from ..components.Load import Load
from ..configurations.config import BatterySpec, CircuitConfig, LoadSpec
from ..configurations.constants import (
    BANK_GAP,
    BASE_FONTSIZE,
    BATTERY_SPACING,
    BUS_RAIL_GAP,
    BUS_SEGMENT,
    COMPONENT_DISTANCE,
    LOAD_SPACING,
    TAG_ROW_GAP,
)
from .mppt_panel_drawing import add_title
from .tags import NEGATIVE, POSITIVE, incoming_tag, mppt_net, tag_width

# Horizontal clearance between adjacent tag bodies.
TAG_CLEARANCE = 0.6

# Clearance kept between a bus bar and the nearest element of a bank.
BUS_CLEARANCE = 1.5

# Horizontal padding added to each end of the bus bars.
BUS_OVERHANG = 1.0

# Leads shorter than this are skipped rather than drawn as zero-length lines.
MIN_LEAD = 1e-9


def _bus_gap(battery: Optional[BatterySpec]) -> float:
    """Vertical separation of the bus bars, widened to clear a tall battery bank."""
    series = battery.in_series if battery else 1
    stack_height = series * COMPONENT_DISTANCE
    return max(BUS_RAIL_GAP, stack_height + 2 * BUS_CLEARANCE)


def _add_string(
    drawing: schemdraw.Drawing,
    element: Type[elm.Element],
    series: int,
    x: float,
    top_y: float,
    label_prefix: str,
    start_index: int,
    element_kwargs: dict,
) -> Point:
    """Draw one vertical string of ``series`` elements downwards from ``(x, top_y)``.

    Returns the bottom end of the string.
    """
    here = Point((x, top_y))
    for offset in range(series):
        component = (
            element(**element_kwargs)
            .down()
            .at(here)
            .label(f'{label_prefix}{start_index + offset}')
        )
        drawing.add(component)
        here = component.end
    return Point(here)


def _tap_bus(drawing: schemdraw.Drawing, point, bus_y: float) -> None:
    """Run a straight vertical lead from ``point`` to ``bus_y`` and dot the junction."""
    if abs(point.y - bus_y) > MIN_LEAD:
        drawing.add(elm.Line().at(point).toy(bus_y))
    drawing.add(elm.Dot().at((point.x, bus_y)))


def _add_bank(
    drawing: schemdraw.Drawing,
    element: Type[elm.Element],
    series: int,
    parallel: int,
    start_x: float,
    spacing: float,
    positive_y: float,
    negative_y: float,
    label_prefix: str,
    element_kwargs: Optional[dict] = None,
    start_index: int = 1,
) -> float:
    """Draw ``parallel`` vertical strings between the bus bars.

    Every string taps both bars directly, so the bars do the paralleling and no
    horizontal rails are needed. Returns the x of the right-most string.
    """
    element_kwargs = element_kwargs or {}

    stack_height = series * COMPONENT_DISTANCE
    centre_y = (positive_y + negative_y) / 2
    top_y = centre_y + stack_height / 2

    index = start_index
    x = start_x
    for string in range(max(1, parallel)):
        x = start_x + string * spacing
        bottom = _add_string(
            drawing, element, series, x, top_y, label_prefix, index, element_kwargs
        )
        _tap_bus(drawing, Point((x, top_y)), positive_y)
        _tap_bus(drawing, bottom, negative_y)
        index += series

    return x


def _add_incoming_tags(
    drawing: schemdraw.Drawing,
    mppt_count: int,
    positive_y: float,
    negative_y: float,
    first_tap_x: float,
) -> list[float]:
    """Add one tag pair per MPPT and drop each onto its bus bar.

    All positive tags share one row above the positive bar and all negative tags
    one row below the negative bar, spaced far enough apart that no two tag
    bodies overlap. Keeping them to a single row per rail costs width but keeps
    the page height independent of the MPPT count, and height is what usually
    limits the scale.

    Returns the tap x positions used.
    """
    # Space taps by the widest tag body so the flags cannot collide. The widest
    # label is the highest index, which may have more digits than the first.
    spacing = max(
        BUS_SEGMENT,
        tag_width(mppt_net(max(mppt_count, 1), POSITIVE)) + TAG_CLEARANCE,
    )

    tap_positions = []
    for index in range(1, mppt_count + 1):
        tap_x = first_tap_x + (index - 1) * spacing
        tap_positions.append(tap_x)

        # Positive tag sits above the positive bus and drops down onto it.
        tag_positive_y = positive_y + TAG_ROW_GAP
        drawing.add(
            incoming_tag(mppt_net(index, POSITIVE), fontsize=BASE_FONTSIZE)
            .at((tap_x, tag_positive_y))
        )
        drawing.add(elm.Line().at((tap_x, tag_positive_y)).toy(positive_y))
        drawing.add(elm.Dot().at((tap_x, positive_y)))

        # Negative tag sits below the negative bus and rises onto it.
        tag_negative_y = negative_y - TAG_ROW_GAP
        drawing.add(
            incoming_tag(mppt_net(index, NEGATIVE), fontsize=BASE_FONTSIZE)
            .at((tap_x, tag_negative_y))
        )
        drawing.add(elm.Line().at((tap_x, tag_negative_y)).toy(negative_y))
        drawing.add(elm.Dot().at((tap_x, negative_y)))

    return tap_positions


def _add_loads(
    drawing: schemdraw.Drawing,
    loads: list[LoadSpec],
    start_x: float,
    positive_y: float,
    negative_y: float,
) -> float:
    """Draw each load as its own single-element string across the bus bars."""
    rightmost = start_x
    for offset, load in enumerate(loads):
        rightmost = _add_bank(
            drawing,
            element=Load,
            series=1,
            parallel=1,
            start_x=start_x + offset * LOAD_SPACING,
            spacing=LOAD_SPACING,
            positive_y=positive_y,
            negative_y=negative_y,
            label_prefix='M',
            start_index=offset + 1,
            # Power only: the load names are listed in the page subtitle, and
            # repeating them on every symbol makes the page much wider.
            element_kwargs={'power_watts': load.total_power},
        )
    return rightmost


def generate_battery_bus_drawing(
    config: CircuitConfig,
    boat_label: Optional[str] = None,
) -> schemdraw.Drawing:
    """Build the shared battery/load bus page.

    Parameters
    ----------
    config:
        Parsed circuit configuration. The number of incoming tag pairs is taken
        from :attr:`CircuitConfig.total_mppt_count`.
    boat_label:
        Optional boat name included in the page subtitle.
    """
    drawing = schemdraw.Drawing()
    drawing.config(unit=COMPONENT_DISTANCE, fontsize=BASE_FONTSIZE)

    battery = config.battery
    gap = _bus_gap(battery)
    positive_y = 0.0
    negative_y = -gap

    mppt_count = config.total_mppt_count

    tap_positions = _add_incoming_tags(drawing, mppt_count, positive_y, negative_y, 0.0)

    tap_end = tap_positions[-1] if tap_positions else 0.0
    rightmost = tap_end

    if battery:
        rightmost = _add_bank(
            drawing,
            element=Battery,
            series=battery.in_series,
            parallel=battery.in_parallel,
            start_x=tap_end + BANK_GAP,
            spacing=BATTERY_SPACING,
            positive_y=positive_y,
            negative_y=negative_y,
            label_prefix='B',
            # Voltage only: the chemistry is named in the page subtitle.
            element_kwargs={'voltage': battery.block_voltage},
        )

    if config.loads:
        rightmost = _add_loads(
            drawing, config.loads, rightmost + BANK_GAP, positive_y, negative_y
        )

    # --- Bus bars, sized to span every tap ---------------------------------
    bus_start_x = min(tap_positions) - BUS_OVERHANG if tap_positions else 0.0
    bus_end_x = rightmost + BUS_OVERHANG
    drawing.add(
        elm.Line()
        .at((bus_start_x, positive_y))
        .to((bus_end_x, positive_y))
        .label('B+', loc='left', fontsize=BASE_FONTSIZE)
    )
    drawing.add(
        elm.Line()
        .at((bus_start_x, negative_y))
        .to((bus_end_x, negative_y))
        .label('B-', loc='left', fontsize=BASE_FONTSIZE)
    )

    # --- Title -------------------------------------------------------------
    parts = [f'{mppt_count} MPPT input{"s" if mppt_count != 1 else ""}']
    if battery:
        parts.append(battery.summary)
    if config.loads:
        parts.append(', '.join(load.display_name for load in config.loads))
    subtitle = '  |  '.join(parts)
    if boat_label:
        subtitle = f'{boat_label}  |  {subtitle}'
    add_title(drawing, 'Battery Bus', subtitle)

    return drawing
