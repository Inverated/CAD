"""Per-MPPT page: solar panel array -> MPPT -> cross-reference tags.

One of these is produced for every MPPT instance in the circuit. The page is
self-contained: it shows the panel array feeding this MPPT and hands the
battery-side connection off to the battery bus page via a pair of tags.
"""

from __future__ import annotations

from typing import Optional

import schemdraw
import schemdraw.elements as elm

from ..Array_Creator import draw_array
from ..components.MPPT import MPPT
from ..configurations.config import MpptSpec, PanelSpec
from ..configurations.constants import (
    BASE_FONTSIZE,
    COMPONENT_DISTANCE,
    MPPT_INPUT_LEAD,
    PANEL_SPACING,
    TAG_LEAD,
    TERMINATING_DISTANCE,
    TITLE_FONTSIZE,
    TITLE_OFFSET,
)
from .tags import NEGATIVE, POSITIVE, bus_net, outgoing_tag

# Pin stub length used by the MPPT element; its PV/BATT anchors sit this far
# outside the body.
MPPT_PIN = 0.5


def add_title(drawing: schemdraw.Drawing, title: str, subtitle: Optional[str] = None) -> None:
    """Add a centred title (and optional subtitle) above the drawing content."""
    bbox = drawing.get_bbox()
    centre_x = (bbox.xmin + bbox.xmax) / 2
    text = title if not subtitle else f'{title}\n{subtitle}'
    drawing.add(
        elm.Label()
        .label(text, fontsize=TITLE_FONTSIZE)
        .at((centre_x, bbox.ymax + TITLE_OFFSET))
    )


def generate_mppt_panel_drawing(
    mppt_index: int,
    panel: PanelSpec,
    mppt: MpptSpec,
    boat_label: Optional[str] = None,
) -> schemdraw.Drawing:
    """Build the page for a single MPPT and its panel array.

    Parameters
    ----------
    mppt_index:
        1-based index of this MPPT across the whole circuit; drives the label
        ("MPPT1") and the tag net names.
    panel:
        Panel array spec (series/parallel already resolved, including any
        boat-parameter fallbacks).
    mppt:
        MPPT model spec.
    boat_label:
        Optional boat name included in the page subtitle.

    Returns
    -------
    schemdraw.Drawing
        Layout, left to right: panel array -> MPPT (PV+/PV-) -> BATT+/BATT-
        -> outgoing tags referencing the battery bus.
    """
    drawing = schemdraw.Drawing()
    drawing.config(unit=COMPONENT_DISTANCE, fontsize=BASE_FONTSIZE)

    mppt_name = f'MPPT{mppt_index}'

    # --- Panel array -----------------------------------------------------
    drawing, panel_terminals = draw_array(
        drawing=drawing,
        element=elm.Solar,
        series=panel.in_series,
        parallel=panel.in_parallel,
        terminateDist=TERMINATING_DISTANCE,
        isRight=True,
        label_prefix='PV',
        spacing=PANEL_SPACING,
    )

    # --- Leads from the array into the MPPT input pins --------------------
    positive_lead = elm.Line().right().at(panel_terminals.positive).length(MPPT_INPUT_LEAD)
    negative_lead = elm.Line().right().at(panel_terminals.negative).length(MPPT_INPUT_LEAD)
    drawing.add(positive_lead)
    drawing.add(negative_lead)

    # --- MPPT, aligned so its PV- pin meets the negative lead -------------
    converter = MPPT(pin_gap=TERMINATING_DISTANCE, name=mppt_name).anchor('PV-').at(negative_lead.end)
    if mppt.choice:
        converter.label(mppt.choice, loc='bottom', fontsize=BASE_FONTSIZE - 1)
    drawing.add(converter)

    # --- Battery-side outputs, handed off to the battery bus page ---------
    for polarity, anchor in ((POSITIVE, 'BATT+'), (NEGATIVE, 'BATT-')):
        lead = elm.Line().right().at(converter.absanchors[anchor]).length(TAG_LEAD)
        drawing.add(lead)
        drawing.add(outgoing_tag(bus_net(polarity), fontsize=BASE_FONTSIZE).at(lead.end))

    # --- Title ------------------------------------------------------------
    subtitle = panel.summary
    if mppt.choice:
        subtitle = f'{subtitle}  |  {mppt.choice}'
    if boat_label:
        subtitle = f'{boat_label}  |  {subtitle}'
    add_title(drawing, f'{mppt_name} — Solar Array', subtitle)

    return drawing
