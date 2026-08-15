"""Cross-reference tags linking the sub-drawings together.

Each connection that crosses a page boundary is drawn as a schemdraw ``Tag``
carrying a net name. Tags come in matched pairs so a reader can follow a net
between pages:

* on an MPPT page:      ``→ Battery Bus B+``   (points at the battery bus page)
* on the battery bus:   ``← MPPT1 BATT+``      (points back at the MPPT page)

Tag geometry must be sized manually because schemdraw cannot measure text, so
:func:`tag_width` estimates a width from the label length. The estimate assumes
the drawing's base font size and scale; the page scaler keeps the text-to-
geometry ratio fixed so these widths stay correct at any page scale.
"""

from __future__ import annotations

import schemdraw.elements as elm

from ..configurations.constants import TAG_CHAR_WIDTH, TAG_MIN_WIDTH, TAG_PADDING

POSITIVE = '+'
NEGATIVE = '-'
POLARITIES = (POSITIVE, NEGATIVE)


def tag_width(text: str) -> float:
    """Estimate the tag body width needed to fit ``text``."""
    return max(TAG_MIN_WIDTH, TAG_CHAR_WIDTH * len(text) + TAG_PADDING)


def mppt_net(mppt_index: int, polarity: str) -> str:
    """Net name for an MPPT's battery-side output, e.g. ``'MPPT1 BATT+'``."""
    return f'MPPT{mppt_index} BATT{polarity}'


def bus_net(polarity: str) -> str:
    """Net name for a battery bus rail, e.g. ``'Battery Bus B+'``."""
    return f'Battery Bus B{polarity}'


def outgoing_tag(text: str, fontsize: float = 10) -> elm.Tag:
    """A tag whose tip points back at the wire, body extending to the right.

    Used on the MPPT pages, where the wire arrives from the left.
    """
    return elm.Tag(width=tag_width(text)).label(f'→ {text}', fontsize=fontsize)


def incoming_tag(text: str, fontsize: float = 10) -> elm.Tag:
    """A tag whose tip points right at the wire, body extending to the left.

    Used on the battery bus page, where the wire continues to the right.
    """
    return elm.Tag(width=tag_width(text)).left().label(f'← {text}', fontsize=fontsize)
