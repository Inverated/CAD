"""Task 4 demo: the shared battery bus page, generated from rp2's config.

rp2 has 2 MPPTs, a 2s1p LiNMC bank and one Torqeedo load, so the page should
show 2 incoming tag pairs, both bus bars, the battery bank and the load. rp3 is
rendered too, to show the tag stack growing with the MPPT count.

Run:  python -m src.electrical_drawing.demos.task4_battery_bus
"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

from ..configurations.config import load_circuit
from ..drawings.battery_bus_drawing import generate_battery_bus_drawing
from .demo_utils import DEMO_DIR, save

COMPONENTS = 'constant/electrical/components.json'


def render(boat: str) -> None:
    config = load_circuit(
        f'constant/electrical/boat/{boat}/circuit_setup.json',
        COMPONENTS,
        f'constant/boat/{boat}.json',
    )
    drawing = generate_battery_bus_drawing(config, boat_label=boat)
    path = save(drawing, f'task4.{boat}_battery_bus')

    print(f'{boat}: {config.total_mppt_count} MPPTs -> '
          f'{2 * config.total_mppt_count} incoming tags')
    print(f'  battery : {config.battery.summary}')
    print(f'  loads   : {[load.display_name for load in config.loads]}')
    print(f'  saved   : {path}')


def main() -> None:
    for boat in ('rp2', 'rp3'):
        render(boat)
    print(f'\nDemo output directory: {DEMO_DIR}')


if __name__ == '__main__':
    main()
