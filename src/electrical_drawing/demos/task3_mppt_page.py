"""Task 3 demo: MPPT + panel array pages generated from rp3's config.

rp3 declares 3 MPPTs with a 4s2p panel array each. This renders MPPT1 and
MPPT2 to prove the generator is driven entirely by configuration.

Run:  python -m src.electrical_drawing.demos.task3_mppt_page
"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

from ..configurations.config import load_circuit
from ..drawings.mppt_panel_drawing import generate_mppt_panel_drawing
from .demo_utils import DEMO_DIR, save

CIRCUIT = 'constant/electrical/boat/rp3/circuit_setup.json'
COMPONENTS = 'constant/electrical/components.json'
BOAT_PARAMS = 'constant/boat/rp3.json'


def main() -> None:
    config = load_circuit(CIRCUIT, COMPONENTS, BOAT_PARAMS)

    print(f'rp3: {config.total_mppt_count} MPPT instances from '
          f'{len(config.mppt_groups)} config block(s)')

    for index, group in config.mppt_instances()[:2]:
        drawing = generate_mppt_panel_drawing(
            mppt_index=index,
            panel=group.panel,
            mppt=group.mppt,
            boat_label='rp3',
        )
        path = save(drawing, f'task3.mppt{index}_panel')
        print(f'  MPPT{index}: {group.panel.summary} -> {path}')

    print(f'\nDemo output directory: {DEMO_DIR}')


if __name__ == '__main__':
    main()
