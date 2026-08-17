"""Load element: a rectangular body representing a motor or other consumer.

Two-terminal element (``start`` / ``end`` anchors supplied by schemdraw's
``Element2Term``), proportioned like ``elm.Motor`` so it sits comfortably
alongside the stock symbols. Compatible with
:func:`~src.electrical_drawing.Array_Creator.draw_array`.

The name/power annotation is applied with ``.label()`` rather than an embedded
text segment so that it stays horizontal even when the element is drawn
downwards.
"""

from __future__ import annotations

from typing import Optional

from schemdraw.elements import Element2Term
from schemdraw.segments import Segment, SegmentCircle

# Body proportions, in drawing units.
BODY_LENGTH = 1.0
BODY_HALF_HEIGHT = 0.45


def format_power(power_watts: Optional[float]) -> str:
    """Render a wattage as kW when >= 1000 W, else as W ('4000' -> '4.0kW')."""
    if power_watts is None:
        return ''
    if abs(power_watts) >= 1000:
        return f'{power_watts / 1000:.1f}kW'
    return f'{power_watts:g}W'


def load_spec(name: Optional[str] = None, power_watts: Optional[float] = None) -> str:
    """Combine a load name and power into a two-line label, skipping missing parts."""
    parts = [part for part in (name, format_power(power_watts)) if part]
    return '\n'.join(parts)


class Load(Element2Term):
    """A rectangular load/motor block.

    Parameters
    ----------
    name:
        Load name, e.g. ``"Torqeedo 4.0"``.
    power_watts:
        Rated power in watts; rendered as kW above 1000 W.
    rotor:
        Draw an inner circle (motor-style) inside the body.
    spec_loc:
        Where to place the spec label relative to the element.
    fontsize:
        Font size for the spec label.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        power_watts: Optional[float] = None,
        rotor: bool = True,
        spec_loc: str = 'bottom',
        fontsize: float = 9,
        **kwargs,
    ):
        super().__init__(**kwargs)

        length = BODY_LENGTH
        half_height = BODY_HALF_HEIGHT

        # Leads: body spans 0..length, schemdraw extends leads from each end.
        self.segments.append(Segment([(0, 0), (0, 0), (length, 0), (length, 0)]))

        # Rectangular body.
        self.segments.append(
            Segment([
                (0, -half_height),
                (length, -half_height),
                (length, half_height),
                (0, half_height),
                (0, -half_height),
            ])
        )

        if rotor:
            self.segments.append(SegmentCircle((length / 2, 0), half_height * 0.55))

        self.name = name
        self.power_watts = power_watts
        self.spec = load_spec(name, power_watts)
        if self.spec:
            self.label(self.spec, loc=spec_loc, fontsize=fontsize)
