"""Task 6 tests: configuration parsing, including optional-field fallbacks."""

from __future__ import annotations

import json

import pytest

from ..configurations.config import (
    load_circuit,
    parse_circuit,
    parse_panel,
)

COMPONENTS = {
    'Panel': {'Renogy_455W': {'power': 455, 'voltage': 41.5}},
    'MPPT': {'Victron_150/45': {'max_input_voltage': 150, 'max_output_current': 45}},
    'Battery': {'LiNMC': {'battery_voltage': 25.9, 'capacity_ah': 100}},
    'Load': {'Torqeedo_Cruise_4.0': {'total_power': 4000, 'nominal_voltage': 48.0}},
}

BOAT_PARAMS = {
    'boat_name': 'rp2',
    'panels_longitudinal': 4,
    'panels_transversal': 2,
    'panels_per_string': 2,
}


def circuit(panel_info=None, mppt_count=2):
    return {
        'display_name': 'Electric Proa',
        'configuration_description': 'RP2',
        'mppt_panel': {
            'config_1': {
                'count': mppt_count,
                'panel_info': panel_info if panel_info is not None else {
                    'choice': 'Renogy_455W', 'in_series': 2, 'in_parallel': 2,
                },
                'mppt_info': {'choice': 'Victron_150/45'},
            }
        },
        'load': {'load_1': {'choice': 'Torqeedo_Cruise_4.0'}},
        'battery': {'choice': 'LiNMC', 'battery_in_series': 2, 'battery_in_parallel': 1},
    }


# --------------------------------------------------------------------------
# Panel fallbacks
# --------------------------------------------------------------------------

def test_explicit_series_and_parallel_are_used():
    panel = parse_panel({'choice': 'Renogy_455W', 'in_series': 4, 'in_parallel': 2},
                        COMPONENTS, BOAT_PARAMS)
    assert (panel.in_series, panel.in_parallel) == (4, 2)


def test_missing_in_series_falls_back_to_panels_per_string():
    panel = parse_panel({'choice': 'Renogy_455W', 'in_parallel': 2}, COMPONENTS, BOAT_PARAMS)
    assert panel.in_series == BOAT_PARAMS['panels_per_string']


def test_missing_in_parallel_is_derived_from_boat_geometry():
    """4 x 2 panels / 2 per string = 4 strings."""
    panel = parse_panel({'choice': 'Renogy_455W', 'in_series': 2}, COMPONENTS, BOAT_PARAMS)
    assert panel.in_parallel == 4


def test_both_missing_uses_both_fallbacks():
    panel = parse_panel({'choice': 'Renogy_455W'}, COMPONENTS, BOAT_PARAMS)
    assert (panel.in_series, panel.in_parallel) == (2, 4)


def test_falls_back_to_one_without_boat_params():
    panel = parse_panel({'choice': 'Renogy_455W'}, COMPONENTS, {})
    assert (panel.in_series, panel.in_parallel) == (1, 1)


def test_unknown_panel_choice_does_not_raise():
    panel = parse_panel({'choice': 'Nonexistent'}, COMPONENTS, BOAT_PARAMS)
    assert panel.power_watts is None
    assert panel.in_series == 2


def test_panel_library_values_are_attached():
    panel = parse_panel({'choice': 'Renogy_455W', 'in_series': 1, 'in_parallel': 1},
                        COMPONENTS, BOAT_PARAMS)
    assert panel.power_watts == 455
    assert panel.voltage == 41.5
    assert panel.total_panels == 1


@pytest.mark.parametrize('bad', [0, -3, None, 'x'])
def test_invalid_counts_fall_back_rather_than_crash(bad):
    panel = parse_panel({'choice': 'Renogy_455W', 'in_series': bad}, COMPONENTS, BOAT_PARAMS)
    assert panel.in_series >= 1


# --------------------------------------------------------------------------
# Whole-circuit parsing
# --------------------------------------------------------------------------

def test_mppt_instances_are_flattened_with_running_index():
    config = parse_circuit(circuit(mppt_count=3), COMPONENTS, BOAT_PARAMS)
    indices = [index for index, _ in config.mppt_instances()]
    assert indices == [1, 2, 3]


def test_total_count_sums_multiple_blocks():
    raw = circuit()
    raw['mppt_panel']['config_2'] = {
        'count': 3,
        'panel_info': {'choice': 'Renogy_455W', 'in_series': 1, 'in_parallel': 1},
        'mppt_info': {'choice': 'Victron_150/45'},
    }
    config = parse_circuit(raw, COMPONENTS, BOAT_PARAMS)
    assert config.total_mppt_count == 5
    assert len(config.mppt_instances()) == 5


def test_battery_totals_and_summary():
    config = parse_circuit(circuit(), COMPONENTS, BOAT_PARAMS)
    assert config.battery.in_series == 2
    assert config.battery.total_voltage == pytest.approx(51.8)
    assert '2s1p' in config.battery.summary


def test_loads_pick_up_library_power():
    config = parse_circuit(circuit(), COMPONENTS, BOAT_PARAMS)
    assert len(config.loads) == 1
    assert config.loads[0].total_power == 4000
    assert config.loads[0].display_name == 'Torqeedo Cruise 4.0'


def test_empty_sections_produce_empty_results():
    config = parse_circuit({}, COMPONENTS, BOAT_PARAMS)
    assert config.mppt_groups == []
    assert config.loads == []
    assert config.total_mppt_count == 0


def test_boat_name_comes_from_boat_params():
    assert parse_circuit(circuit(), COMPONENTS, BOAT_PARAMS).boat_name == 'rp2'


# --------------------------------------------------------------------------
# Loading from disk
# --------------------------------------------------------------------------

def test_load_circuit_reads_files(tmp_path):
    circuit_path = tmp_path / 'circuit.json'
    components_path = tmp_path / 'components.json'
    boat_path = tmp_path / 'boat.json'
    circuit_path.write_text(json.dumps(circuit()))
    components_path.write_text(json.dumps(COMPONENTS))
    boat_path.write_text(json.dumps(BOAT_PARAMS))

    config = load_circuit(str(circuit_path), str(components_path), str(boat_path))
    assert config.total_mppt_count == 2
    assert config.boat_name == 'rp2'


def test_load_circuit_without_boat_params(tmp_path):
    circuit_path = tmp_path / 'circuit.json'
    components_path = tmp_path / 'components.json'
    circuit_path.write_text(json.dumps(circuit()))
    components_path.write_text(json.dumps(COMPONENTS))

    config = load_circuit(str(circuit_path), str(components_path), None)
    assert config.boat_name is None
    assert config.total_mppt_count == 2


# --------------------------------------------------------------------------
# Real repository configuration
# --------------------------------------------------------------------------

@pytest.mark.parametrize('boat,expected_mppts', [('rp1', 2), ('rp2', 2), ('rp3', 3)])
def test_repository_configs_report_expected_mppt_counts(boat, expected_mppts):
    config = load_circuit(
        f'constant/electrical/boat/{boat}/circuit_setup.json',
        'constant/electrical/components.json',
        f'constant/boat/{boat}.json',
    )
    assert config.total_mppt_count == expected_mppts
    assert len(config.loads) == 1
