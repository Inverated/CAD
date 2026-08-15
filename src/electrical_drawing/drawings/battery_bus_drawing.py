"""Battery bus page: every MPPT output, the battery bank and the loads.

This is the single shared page that all MPPT pages point at. Two horizontal bus
bars run across it — positive on top, negative below — and everything taps onto
them:

* incoming tags on the left, one pair per MPPT instance, each dropping onto its
  bus bar through a vertical lead with a junction dot
* the battery bank, wired between the two bars
* each load, wired between the two bars in parallel with the battery

Bus bars are drawn last so they can be sized to span every tap point.
"""

from __future__ import annotations

from typing import Optional

import schemdraw
import schemdraw.elements as elm
from schemdraw.util import Point

from ..Array_Creator import draw_array
from ..components.Battery import Battery
from ..components.Load import Load
from ..configurations.config import BatterySpec, CircuitConfig, LoadSpec
from ..configurations.constants import (
    BASE_FONTSIZE,
    BATTERY_SPACING,
    BUS_RAIL_GAP,
    BUS_SEGMENT,
    COMPONENT_DISTANCE,
    LOAD_RAIL,
    LOAD_SPACING,
)
from .mppt_panel_drawing import add_title
from .tags import NEGATIVE, POSITIVE, incoming_tag, mppt_net

# Vertical spacing between successive tag rows outside the bus bars.
TAG_ROW_GAP = 1.6

# Clearance kept between a bus bar and the array wired to it.
BUS_CLEARANCE = 1.5

# Horizontal padding added to each end of the bus bars.
BUS_OVERHANG = 1.0


def _bus_gap(battery: Optional[BatterySpec]) -> float:
    """Vertical separation of the bus bars, widened to clear a tall battery bank."""
    series = battery.in_series if battery else 1
    stack_height = series * COMPONENT_DISTANCE
    return max(BUS_RAIL_GAP, stack_height + 2 * BUS_CLEARANCE)


def _connect_to_buses(
    drawing: schemdraw.Drawing,
    terminals,
    positive_y: float,
    negative_y: float,
) -> None:
    """Run each terminal of an array vertically onto its bus bar."""
    drawing.add(elm.Line().at(terminals.positive).toy(positive_y))
    drawing.add(elm.Dot().at((terminals.positive.x, positive_y)))
    drawing.add(elm.Line().at(terminals.negative).toy(negative_y))
    drawing.add(elm.Dot().at((terminals.negative.x, negative_y)))


def _add_incoming_tags(
    drawing: schemdraw.Drawing,
    mppt_count: int,
    positive_y: float,
    negative_y: float,
    first_tap_x: float,
) -> list[float]:
    """Add one tag pair per MPPT and drop each onto its bus bar.

    Successive MPPTs are placed further right *and* further from the bus bars, so
    no vertical lead ever crosses another tag body.

    Returns the tap x positions used.
    """
    tap_positions = []
    for index in range(1, mppt_count + 1):
        tap_x = first_tap_x + (index - 1) * BUS_SEGMENT
        tap_positions.append(tap_x)
        offset = index * TAG_ROW_GAP

        # Positive tag sits above the positive bus and drops down onto it.
        tag_positive_y = positive_y + offset
        drawing.add(
            incoming_tag(mppt_net(index, POSITIVE), fontsize=BASE_FONTSIZE)
            .at((tap_x, tag_positive_y))
        )
        drawing.add(elm.Line().at((tap_x, tag_positive_y)).toy(positive_y))
        drawing.add(elm.Dot().at((tap_x, positive_y)))

        # Negative tag sits below the negative bus and rises onto it.
        tag_negative_y = negative_y - offset
        drawing.add(
            incoming_tag(mppt_net(index, NEGATIVE), fontsize=BASE_FONTSIZE)
            .at((tap_x, tag_negative_y))
        )
        drawing.add(elm.Line().at((tap_x, tag_negative_y)).toy(negative_y))
        drawing.add(elm.Dot().at((tap_x, negative_y)))

    return tap_positions


def _add_battery_bank(
    drawing: schemdraw.Drawing,
    battery: BatterySpec,
    start_x: float,
    positive_y: float,
    negative_y: float,
    gap: float,
):
    """Draw the battery bank between the bus bars and return its terminals."""
    stack_height = battery.in_series * COMPONENT_DISTANCE
    top_y = positive_y - (gap - stack_height) / 2

    drawing.here = Point((start_x, top_y))
    drawing, terminals = draw_array(
        drawing=drawing,
        element=Battery,
        series=battery.in_series,
        parallel=battery.in_parallel,
        label_prefix='B',
        spacing=BATTERY_SPACING,
        # Request the array's natural terminal gap so no dogleg is inserted; the
        # vertical leads to the bus bars do the routing instead.
        terminateDist=stack_height,
        element_kwargs={
            'voltage': battery.block_voltage,
            'chemistry': battery.choice,
        },
    )
    _connect_to_buses(drawing, terminals, positive_y, negative_y)
    return terminals


def _add_loads(
    drawing: schemdraw.Drawing,
    loads: list[LoadSpec],
    start_x: float,
    positive_y: float,
    negative_y: float,
    gap: float,
) -> float:
    """Draw each load in parallel across the bus bars.

    Returns the right-most x reached.
    """
    stack_height = COMPONENT_DISTANCE
    top_y = positive_y - (gap - stack_height) / 2
    rightmost = start_x

    for offset, load in enumerate(loads):
        drawing.here = Point((start_x + offset * LOAD_SPACING, top_y))
        drawing, terminals = draw_array(
            drawing=drawing,
            element=Load,
            series=1,
            parallel=1,
            label_prefix='M',
            start_index=offset + 1,
            spacing=LOAD_RAIL,
            terminateDist=stack_height,
            element_kwargs={
                'name': load.display_name,
                'power_watts': load.total_power,
            },
        )
        _connect_to_buses(drawing, terminals, positive_y, negative_y)
        rightmost = max(rightmost, terminals.positive.x)

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
    first_tap_x = 0.0

    tap_positions = _add_incoming_tags(
        drawing, mppt_count, positive_y, negative_y, first_tap_x
    )

    tap_end = tap_positions[-1] if tap_positions else first_tap_x
    battery_x = tap_end + BUS_SEGMENT

    rightmost = battery_x
    if battery:
        battery_terminals = _add_battery_bank(
            drawing, battery, battery_x, positive_y, negative_y, gap
        )
        rightmost = battery_terminals.positive.x

    if config.loads:
        rightmost = _add_loads(
            drawing, config.loads, rightmost + LOAD_SPACING, positive_y, negative_y, gap
        )

    # --- Bus bars, sized to span every tap ---------------------------------
    bus_start_x = min(tap_positions) - BUS_OVERHANG if tap_positions else first_tap_x
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
