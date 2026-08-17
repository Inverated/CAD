"""Task 1 demo: structured terminal output from draw_array().

Draws a 2s2p battery array, prints the returned TerminalPair (positions and
labels) and saves the drawing so the geometry can be eyeballed.

Run:  python -m src.electrical_drawing.demos.task1_array
"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import schemdraw
import schemdraw.elements as elm

from ..Array_Creator import component_labels, draw_array
from ..configurations.constants import TERMINATING_DISTANCE
from .demo_utils import DEMO_DIR, save


def main() -> None:
    drawing = schemdraw.Drawing()
    drawing.config(unit=2)

    drawing, terminals = draw_array(
        drawing=drawing,
        element=elm.Battery,
        series=2,
        parallel=2,
        terminateDist=TERMINATING_DISTANCE,
        isRight=True,
        label_prefix='B',
        element_label='12V',
    )

    print('2s2p battery array')
    print(f'  positive terminal : {tuple(round(v, 3) for v in terminals.positive)}'
          f'  label={terminals.positive_label!r}')
    print(f'  negative terminal : {tuple(round(v, 3) for v in terminals.negative)}'
          f'  label={terminals.negative_label!r}')
    print(f'  terminal gap      : {terminals.gap} (expected {TERMINATING_DISTANCE})')
    print(f'  component labels  : {component_labels("B", 2, 2, "12V")}')

    path = save(drawing, 'task1.battery_array_2s2p')
    print(f'\nSaved {path}')
    print(f'Demo output directory: {DEMO_DIR}')


if __name__ == '__main__':
    main()
