"""Task 2 demo: Battery and Load custom elements inside draw_array().

Draws a 2s1p battery array labelled "25.9V LiNMC" and a 1s2p load array
labelled "Torqeedo 4.0kW", proving both custom elements work as the ``element``
argument to ``draw_array``.

Run:  python -m src.electrical_drawing.demos.task2_elements
"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import schemdraw

from ..Array_Creator import draw_array
from ..components.Battery import Battery
from ..components.Load import Load
from ..configurations.constants import TERMINATING_DISTANCE
from .demo_utils import DEMO_DIR, save


def battery_array() -> str:
    drawing = schemdraw.Drawing()
    drawing.config(unit=2)
    drawing, terminals = draw_array(
        drawing=drawing,
        element=Battery,
        series=2,
        parallel=1,
        terminateDist=TERMINATING_DISTANCE,
        label_prefix='B',
        element_kwargs={'voltage': 25.9, 'chemistry': 'LiNMC'},
    )
    print('2s1p battery array (25.9V LiNMC)')
    print(f'  {terminals.positive_label} at {tuple(round(v, 3) for v in terminals.positive)}')
    print(f'  {terminals.negative_label} at {tuple(round(v, 3) for v in terminals.negative)}')
    return save(drawing, 'task2.battery_array_2s1p')


def load_array() -> str:
    drawing = schemdraw.Drawing()
    drawing.config(unit=2)
    drawing, terminals = draw_array(
        drawing=drawing,
        element=Load,
        series=1,
        parallel=2,
        terminateDist=TERMINATING_DISTANCE,
        label_prefix='M',
        element_kwargs={'name': 'Torqeedo', 'power_watts': 4000},
        spacing=4,
    )
    print('1s2p load array (Torqeedo 4.0kW)')
    print(f'  {terminals.positive_label} at {tuple(round(v, 3) for v in terminals.positive)}')
    print(f'  {terminals.negative_label} at {tuple(round(v, 3) for v in terminals.negative)}')
    return save(drawing, 'task2.load_array_1s2p')


def main() -> None:
    print(f'Saved {battery_array()}')
    print(f'Saved {load_array()}')
    print(f'\nDemo output directory: {DEMO_DIR}')


if __name__ == '__main__':
    main()
