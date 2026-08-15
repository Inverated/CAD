"""Task 6 tests: orchestrator and CLI, end to end."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use('Agg')

import pytest

from ..__main__ import build_parser, main
from ..configurations.config import CircuitConfig, load_circuit
from ..drawing_generator import build_drawings, generate_all, infer_boat_name
from ..output.pdf_compiler import page_count

COMPONENTS = 'constant/electrical/components.json'

# rp1 and rp2 have 2 MPPTs, rp3 has 3; every boat adds one battery bus page.
BOAT_PAGE_COUNTS = [('rp1', 3), ('rp2', 3), ('rp3', 4)]


def paths_for(boat: str) -> dict:
    return {
        'circuit_path': f'constant/electrical/boat/{boat}/circuit_setup.json',
        'components_path': COMPONENTS,
        'boat_params_path': f'constant/boat/{boat}.json',
    }


# --------------------------------------------------------------------------
# Page composition
# --------------------------------------------------------------------------

@pytest.mark.parametrize('boat,expected_pages', BOAT_PAGE_COUNTS)
def test_page_count_is_one_per_mppt_plus_the_bus(boat, expected_pages):
    config = load_circuit(**{
        'circuit_path': paths_for(boat)['circuit_path'],
        'components_path': COMPONENTS,
        'boat_params_path': paths_for(boat)['boat_params_path'],
    })
    pages = build_drawings(config, boat)
    assert len(pages) == expected_pages
    assert len(pages) == config.total_mppt_count + 1


@pytest.mark.parametrize('boat,expected_pages', BOAT_PAGE_COUNTS)
def test_page_names_follow_the_documented_scheme(boat, expected_pages):
    config = load_circuit(
        paths_for(boat)['circuit_path'], COMPONENTS, paths_for(boat)['boat_params_path']
    )
    names = [name for name, _ in build_drawings(config, boat)]
    for index in range(1, expected_pages):
        assert f'{boat}.mppt{index}_panel' in names
    assert f'{boat}.battery_bus' in names
    assert names[-1] == f'{boat}.battery_bus'


# --------------------------------------------------------------------------
# Boat name inference
# --------------------------------------------------------------------------

def test_boat_name_prefers_the_config():
    config = load_circuit(paths_for('rp3')['circuit_path'], COMPONENTS, 'constant/boat/rp3.json')
    assert infer_boat_name(config, output_path='artifact/whatever') == 'rp3'


def test_boat_name_falls_back_to_the_output_directory():
    blank = CircuitConfig(boat_name=None, display_name=None, description=None)
    assert infer_boat_name(blank, output_path='artifact/rp2.electrical_drawing') == 'rp2'


def test_boat_name_falls_back_to_the_circuit_path():
    blank = CircuitConfig(boat_name=None, display_name=None, description=None)
    inferred = infer_boat_name(blank, circuit_path='constant/electrical/boat/rp1/circuit_setup.json')
    assert inferred == 'rp1'


def test_boat_name_has_a_final_default():
    blank = CircuitConfig(boat_name=None, display_name=None, description=None)
    assert infer_boat_name(blank) == 'boat'


# --------------------------------------------------------------------------
# End-to-end output
# --------------------------------------------------------------------------

@pytest.mark.parametrize('boat,expected_pages', BOAT_PAGE_COUNTS)
def test_generate_all_writes_expected_tree(tmp_path, boat, expected_pages):
    output = tmp_path / f'{boat}.electrical_drawing'
    result = generate_all(output_path=str(output), **paths_for(boat))

    # PDF at the top level, named after the boat.
    pdf = output / f'{boat}.electrical_drawing.pdf'
    assert pdf.exists() and pdf.stat().st_size > 0
    assert result.pdf_path == str(pdf)
    assert page_count(str(pdf)) == expected_pages

    # One SVG per page in the pages/ subfolder.
    pages_dir = output / 'pages'
    assert pages_dir.is_dir()
    svgs = sorted(os.listdir(pages_dir))
    assert len(svgs) == expected_pages
    assert all(name.endswith('.svg') for name in svgs)
    assert f'{boat}.battery_bus.svg' in svgs


@pytest.mark.parametrize('boat,expected_pages', BOAT_PAGE_COUNTS)
def test_every_page_fits_the_page_size(tmp_path, boat, expected_pages):
    result = generate_all(
        output_path=str(tmp_path / boat), **paths_for(boat)
    )
    assert len(result.fits) == expected_pages
    for name, fit in zip(result.page_names, result.fits):
        assert fit.fits(), f'{name} overflows the page'


def test_extra_page_formats_are_written(tmp_path):
    result = generate_all(
        output_path=str(tmp_path / 'rp2'),
        page_formats=('svg', 'png'),
        **paths_for('rp2'),
    )
    written = os.listdir(os.path.join(str(tmp_path / 'rp2'), 'pages'))
    assert sum(name.endswith('.svg') for name in written) == result.page_count
    assert sum(name.endswith('.png') for name in written) == result.page_count


def test_custom_page_size_is_applied(tmp_path):
    result = generate_all(
        output_path=str(tmp_path / 'rp2'),
        page_width_inches=8.27,
        page_height_inches=11.69,
        **paths_for('rp2'),
    )
    for fit in result.fits:
        assert fit.page_width_inches == pytest.approx(8.27)
        assert fit.fits()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_parser_requires_the_documented_arguments():
    parser = build_parser()
    args = parser.parse_args([
        '--circuit', 'c.json', '--components', 'k.json',
        '--boat-params', 'b.json', '--output', 'out',
    ])
    assert args.circuit == 'c.json'
    assert args.components == 'k.json'
    assert args.boat_params == 'b.json'
    assert args.output == 'out'


@pytest.mark.parametrize('missing', ['--circuit', '--components', '--boat-params', '--output'])
def test_parser_rejects_missing_required_arguments(missing):
    full = {
        '--circuit': 'c.json', '--components': 'k.json',
        '--boat-params': 'b.json', '--output': 'out',
    }
    argv = []
    for flag, value in full.items():
        if flag != missing:
            argv += [flag, value]
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_cli_runs_end_to_end(tmp_path, capsys):
    output = tmp_path / 'rp2.electrical_drawing'
    exit_code = main([
        '--circuit', 'constant/electrical/boat/rp2/circuit_setup.json',
        '--components', COMPONENTS,
        '--boat-params', 'constant/boat/rp2.json',
        '--output', str(output),
    ])
    assert exit_code == 0

    printed = capsys.readouterr().out
    assert 'rp2' in printed
    assert '3 pages' in printed
    assert (output / 'rp2.electrical_drawing.pdf').exists()
    assert page_count(str(output / 'rp2.electrical_drawing.pdf')) == 3
