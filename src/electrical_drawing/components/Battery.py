"""Battery element with a configurable spec label.

Thin wrapper over schemdraw's battery symbol that adds a formatted
voltage/chemistry annotation. Two-terminal, so it drops straight into
:func:`~src.electrical_drawing.Array_Creator.draw_array`.

The spec label is placed on the opposite side to schemdraw's default label
position, leaving the default side free for the designator that ``draw_array``
applies (e.g. ``B3``).
"""

from __future__ import annotations

from typing import Optional

import schemdraw.elements as elm


def format_voltage(voltage: Optional[float]) -> str:
    """Render a voltage without a trailing ``.0`` (25.9 -> '25.9', 12.0 -> '12')."""
    if voltage is None:
        return ''
    if float(voltage).is_integer():
        return f'{int(voltage)}V'
    return f'{voltage:g}V'


def battery_spec(voltage: Optional[float] = None, chemistry: Optional[str] = None) -> str:
    """Combine voltage and chemistry into a single label, skipping missing parts."""
    parts = [part for part in (format_voltage(voltage), chemistry) if part]
    return ' '.join(parts)


class Battery(elm.Battery):
    """A battery cell/block annotated with its voltage and chemistry.

    Parameters
    ----------
    voltage:
        Nominal voltage of this block, e.g. ``25.9``.
    chemistry:
        Chemistry or model name, e.g. ``"LiNMC"``.
    spec_loc:
        Where to place the spec label relative to the element.
    fontsize:
        Font size for the spec label.
    """

    def __init__(
        self,
        voltage: Optional[float] = None,
        chemistry: Optional[str] = None,
        spec_loc: str = 'bottom',
        fontsize: float = 9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.voltage = voltage
        self.chemistry = chemistry
        self.spec = battery_spec(voltage, chemistry)
        if self.spec:
            self.label(self.spec, loc=spec_loc, fontsize=fontsize)
