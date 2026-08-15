"""Task 2 tests: Battery and Load custom elements."""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import pytest
import schemdraw

from ..Array_Creator import draw_array
from ..components.Battery import Battery, battery_spec, format_voltage
from ..components.Load import Load, format_power, load_spec
from ..components.MPPT import MPPT

UNIT = 2


def place(element):
    """Add ``element`` to a fresh drawing pointing right, and return it placed."""
    drawing = schemdraw.Drawing()
    drawing.config(unit=UNIT)
    drawing.add(element.right())
    return element


# --------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------

@pytest.mark.parametrize('element', [Battery(), Load()])
def test_two_terminal_anchors_span_one_unit(element):
    placed = place(element)
    assert placed.start.x == pytest.approx(0)
    assert placed.start.y == pytest.approx(0)
    assert placed.end.x == pytest.approx(UNIT)
    assert placed.end.y == pytest.approx(0)


@pytest.mark.parametrize('element', [Battery(), Load()])
def test_center_is_midway_between_terminals(element):
    placed = place(element)
    assert placed.center.x == pytest.approx(UNIT / 2)
    assert placed.center.y == pytest.approx(0)


def test_load_body_is_rectangular():
    """The body outline is a closed 4-corner rectangle."""
    load = Load(rotor=False)
    body = load.segments[1]
    xs = {round(point[0], 6) for point in body.path}
    ys = {round(point[1], 6) for point in body.path}
    assert len(xs) == 2
    assert len(ys) == 2


def test_load_rotor_adds_a_circle():
    assert len(Load(rotor=True).segments) == len(Load(rotor=False).segments) + 1


def test_mppt_exposes_four_terminals():
    mppt = MPPT(pin_gap=2)
    for anchor in ('PV+', 'PV-', 'BATT+', 'BATT-'):
        assert anchor in mppt.anchors


# --------------------------------------------------------------------------
# Label formatting
# --------------------------------------------------------------------------

def test_format_voltage_strips_trailing_zero():
    assert format_voltage(12.0) == '12V'
    assert format_voltage(25.9) == '25.9V'
    assert format_voltage(None) == ''


def test_battery_spec_combines_parts_and_skips_missing():
    assert battery_spec(25.9, 'LiNMC') == '25.9V LiNMC'
    assert battery_spec(None, 'LiNMC') == 'LiNMC'
    assert battery_spec(12, None) == '12V'
    assert battery_spec(None, None) == ''


def test_format_power_switches_to_kilowatts():
    assert format_power(4000) == '4.0kW'
    assert format_power(250) == '250W'
    assert format_power(None) == ''


def test_load_spec_is_two_lines():
    assert load_spec('Torqeedo', 4000) == 'Torqeedo\n4.0kW'
    assert load_spec('Torqeedo', None) == 'Torqeedo'
    assert load_spec(None, None) == ''


def test_battery_stores_spec_on_instance():
    battery = Battery(voltage=25.9, chemistry='LiNMC')
    assert battery.spec == '25.9V LiNMC'


def test_load_stores_spec_on_instance():
    load = Load(name='Torqeedo', power_watts=4000)
    assert load.spec == 'Torqeedo\n4.0kW'


# --------------------------------------------------------------------------
# draw_array compatibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize('series,parallel', [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize(
    'element,kwargs',
    [
        (Battery, {'voltage': 25.9, 'chemistry': 'LiNMC'}),
        (Load, {'name': 'Torqeedo', 'power_watts': 4000}),
    ],
)
def test_custom_elements_pass_through_draw_array(element, kwargs, series, parallel):
    drawing = schemdraw.Drawing()
    drawing.config(unit=UNIT)
    drawing, terminals = draw_array(
        drawing=drawing,
        element=element,
        series=series,
        parallel=parallel,
        element_kwargs=kwargs,
    )
    assert terminals.gap == pytest.approx(2)
    assert terminals.positive.y > terminals.negative.y


def test_custom_elements_render_without_error(tmp_path):
    drawing = schemdraw.Drawing()
    drawing.config(unit=UNIT)
    draw_array(drawing=drawing, element=Battery, series=2, parallel=1,
               element_kwargs={'voltage': 25.9, 'chemistry': 'LiNMC'})
    out = tmp_path / 'battery.svg'
    drawing.save(str(out))
    assert out.exists() and out.stat().st_size > 0
