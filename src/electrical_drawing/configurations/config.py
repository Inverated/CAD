"""Parsing of the electrical JSON configuration into drawing-ready structures.

Reads ``circuit_setup.json`` (per boat), ``components.json`` (the component
library) and optionally the boat parameter file, and exposes a single
:class:`CircuitConfig` describing everything the drawing generators need.

All component-library lookups are tolerant: a missing entry yields ``None``
fields rather than raising, so a drawing can still be produced for a partially
specified configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


def load_json(path: str) -> dict:
    """Load a JSON file into a dict."""
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _lookup(components: dict, category: str, choice: Optional[str]) -> dict:
    """Fetch ``components[category][choice]``, returning ``{}`` when absent."""
    if not choice:
        return {}
    entry = components.get(category, {}).get(choice)
    return entry if isinstance(entry, dict) else {}


@dataclass(frozen=True)
class PanelSpec:
    """A panel array feeding one MPPT."""

    choice: Optional[str]
    in_series: int
    in_parallel: int
    power_watts: Optional[float] = None
    voltage: Optional[float] = None

    @property
    def total_panels(self) -> int:
        return self.in_series * self.in_parallel

    @property
    def summary(self) -> str:
        """Human-readable spec, e.g. '4s2p Renogy_455W (8 x 455W)'."""
        text = f'{self.in_series}s{self.in_parallel}p'
        if self.choice:
            text = f'{text} {self.choice}'
        if self.power_watts:
            text = f'{text} ({self.total_panels} x {self.power_watts:g}W)'
        return text


@dataclass(frozen=True)
class MpptSpec:
    """One MPPT model."""

    choice: Optional[str]
    max_input_voltage: Optional[float] = None
    max_output_current: Optional[float] = None


@dataclass(frozen=True)
class MpptGroup:
    """A repeated MPPT+panel configuration block (``config_N``)."""

    name: str
    count: int
    panel: PanelSpec
    mppt: MpptSpec


@dataclass(frozen=True)
class BatterySpec:
    """The battery bank."""

    choice: Optional[str]
    in_series: int
    in_parallel: int
    block_voltage: Optional[float] = None
    capacity_ah: Optional[float] = None

    @property
    def total_voltage(self) -> Optional[float]:
        if self.block_voltage is None:
            return None
        return self.block_voltage * self.in_series

    @property
    def summary(self) -> str:
        text = f'{self.in_series}s{self.in_parallel}p'
        if self.choice:
            text = f'{text} {self.choice}'
        if self.total_voltage:
            text = f'{text} ({self.total_voltage:g}V nominal)'
        return text


@dataclass(frozen=True)
class LoadSpec:
    """One load / motor entry."""

    name: str
    choice: Optional[str]
    total_power: Optional[float] = None
    nominal_voltage: Optional[float] = None

    @property
    def display_name(self) -> str:
        return (self.choice or self.name).replace('_', ' ')


@dataclass(frozen=True)
class CircuitConfig:
    """Everything needed to draw a boat's electrical schematic."""

    boat_name: Optional[str]
    display_name: Optional[str]
    description: Optional[str]
    mppt_groups: list[MpptGroup] = field(default_factory=list)
    loads: list[LoadSpec] = field(default_factory=list)
    battery: Optional[BatterySpec] = None

    @property
    def total_mppt_count(self) -> int:
        """Total number of MPPT instances across every ``config_N`` block."""
        return sum(group.count for group in self.mppt_groups)

    def mppt_instances(self) -> list[tuple[int, MpptGroup]]:
        """Flatten the groups into ``(1-based mppt index, group)`` pairs."""
        instances = []
        index = 1
        for group in self.mppt_groups:
            for _ in range(group.count):
                instances.append((index, group))
                index += 1
        return instances


def _positive_int(value: Any, fallback: int = 1) -> int:
    """Coerce to an int >= 1, falling back when missing or unusable."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return max(1, fallback)
    return number if number >= 1 else max(1, fallback)


def _derive_parallel(boat_params: dict) -> Optional[int]:
    """Derive panel strings from the boat geometry, if fully specified.

    ``panels_longitudinal * panels_transversal / panels_per_string``
    """
    longitudinal = boat_params.get('panels_longitudinal')
    transversal = boat_params.get('panels_transversal')
    per_string = boat_params.get('panels_per_string')
    if not (longitudinal and transversal and per_string):
        return None
    strings = (longitudinal * transversal) / per_string
    return int(strings) if strings >= 1 else None


def parse_panel(panel_info: dict, components: dict, boat_params: dict) -> PanelSpec:
    """Build a :class:`PanelSpec`, applying boat-parameter fallbacks.

    ``in_series`` falls back to ``panels_per_string``; ``in_parallel`` falls back
    to the geometry-derived string count. Both ultimately default to 1.
    """
    choice = panel_info.get('choice')
    library = _lookup(components, 'Panel', choice)

    in_series = _positive_int(
        panel_info.get('in_series'),
        fallback=_positive_int(boat_params.get('panels_per_string'), 1),
    )
    in_parallel = _positive_int(
        panel_info.get('in_parallel'),
        fallback=_positive_int(_derive_parallel(boat_params), 1),
    )

    return PanelSpec(
        choice=choice,
        in_series=in_series,
        in_parallel=in_parallel,
        power_watts=library.get('power'),
        voltage=library.get('voltage'),
    )


def parse_mppt(mppt_info: dict, components: dict) -> MpptSpec:
    choice = mppt_info.get('choice')
    library = _lookup(components, 'MPPT', choice)
    return MpptSpec(
        choice=choice,
        max_input_voltage=library.get('max_input_voltage'),
        max_output_current=library.get('max_output_current'),
    )


def parse_battery(battery_info: dict, components: dict) -> BatterySpec:
    choice = battery_info.get('choice')
    library = _lookup(components, 'Battery', choice)
    return BatterySpec(
        choice=choice,
        in_series=_positive_int(battery_info.get('battery_in_series'), 1),
        in_parallel=_positive_int(battery_info.get('battery_in_parallel'), 1),
        block_voltage=library.get('battery_voltage'),
        capacity_ah=library.get('capacity_ah'),
    )


def parse_loads(load_section: dict, components: dict) -> list[LoadSpec]:
    loads = []
    for name in sorted(load_section.keys()):
        entry = load_section[name]
        if not isinstance(entry, dict):
            continue
        choice = entry.get('choice')
        library = _lookup(components, 'Load', choice)
        loads.append(
            LoadSpec(
                name=name,
                choice=choice,
                total_power=library.get('total_power'),
                nominal_voltage=library.get('nominal_voltage'),
            )
        )
    return loads


def parse_circuit(circuit: dict, components: dict, boat_params: Optional[dict] = None) -> CircuitConfig:
    """Build a :class:`CircuitConfig` from already-loaded dicts."""
    boat_params = boat_params or {}

    mppt_groups = []
    for name in sorted(circuit.get('mppt_panel', {}).keys()):
        block = circuit['mppt_panel'][name]
        if not isinstance(block, dict):
            continue
        mppt_groups.append(
            MpptGroup(
                name=name,
                count=_positive_int(block.get('count'), 1),
                panel=parse_panel(block.get('panel_info', {}), components, boat_params),
                mppt=parse_mppt(block.get('mppt_info', {}), components),
            )
        )

    return CircuitConfig(
        boat_name=boat_params.get('boat_name'),
        display_name=circuit.get('display_name'),
        description=circuit.get('configuration_description'),
        mppt_groups=mppt_groups,
        loads=parse_loads(circuit.get('load', {}), components),
        battery=parse_battery(circuit.get('battery', {}), components),
    )


def load_circuit(
    circuit_path: str,
    components_path: str,
    boat_params_path: Optional[str] = None,
) -> CircuitConfig:
    """Load and parse all configuration files from disk."""
    circuit = load_json(circuit_path)
    components = load_json(components_path)
    boat_params = load_json(boat_params_path) if boat_params_path else {}
    return parse_circuit(circuit, components, boat_params)
