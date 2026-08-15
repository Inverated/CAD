# Electrical Drawing Module (`src/electrical_drawing`)

## Purpose

Generates schematic circuit diagrams (2D vector drawings) of the solar proa's electrical system using the `schemdraw` library. Reads the same JSON configuration files used by `electrical_simulation` and produces visual schematics showing power flow from solar panels through MPPTs to batteries and loads.

## Status: Early Development

Basic file structure created. Array drawing and MPPT component exist as proof-of-concept. Full configuration-driven generation is planned.

---

## Architecture (Current)

```
__main__.py              Entry point (currently hardcoded demo)
├── Array_Creator.py     Draws series/parallel component arrays
├── components/
│   └── MPPT.py          Custom schemdraw element (PV+/- → BATT+/-)
└── configurations/
    └── constants.py     Layout constants (COMPONENT_DISTANCE, TERMINATING_DISTANCE)
```

### Architecture (Planned)

```
__main__.py              CLI entry point with argparse
├── drawing_generator.py Orchestrator: load config → build schematic → save
├── Array_Creator.py     Draws series/parallel arrays (panels, batteries)
├── components/
│   ├── MPPT.py          MPPT charge controller block
│   ├── Load.py          Motor/load block
│   └── Battery.py       Battery element with labeling
└── configurations/
    └── constants.py     Layout spacing constants
```

---

## Dependencies (Centralised Parameters)

Will consume the same configuration files as `electrical_simulation`:

| File | Purpose | Key Fields Used |
|------|---------|----------------|
| `constant/electrical/{boat}_circuit_setup.json` | Circuit topology | `mppt_panel` (arrays, count, choices), `load`, `battery` (series/parallel) |
| `constant/electrical/electrical_components.json` | Component specs | Panel power/voltage, MPPT limits, Battery voltage, Load power |
| `constant/boat/{boat}.json` | Panel layout | `panels_per_string` → series count, `panels_longitudinal / panels_transversal` → parallel count |

### Dependency Graph (Planned)

```
constant/boat/{boat}.json ──────────────────┐
constant/electrical/{boat}_circuit_setup.json ├──→ drawing_generator → schemdraw → SVG/PNG
constant/electrical/electrical_components.json ┘
                                                         │
                                                         ▼
                                              artifact/{boat}.electrical_drawing.svg
                                              artifact/{boat}.electrical_drawing.png
```

---

## CLI Usage (Planned)

```bash
python -m src.electrical_drawing \
  --circuit constant/electrical/rp2_circuit_setup.json \
  --components constant/electrical/electrical_components.json \
  --boat-params constant/boat/rp2.json \
  --output artifact/rp2.electrical_drawing
```

### Makefile Integration (Planned)

```makefile
make electrical-drawing BOAT=rp2
```

---

## Existing Components

### Array_Creator (`draw_array()`)

Draws a series/parallel arrangement of any schemdraw element:
- **Input**: element type, series count, parallel count, terminating distance
- **Output**: drawing with upper/lower terminal wires
- **Logic**: Places elements vertically (series), connects horizontally (parallel), merges terminals to consistent spacing

### MPPT Component

Custom `schemdraw.Element` with:
- 4 pins: PV+, PV-, BATT+, BATT-
- Rectangular body with "MPPT" label
- Configurable pin gap (matches TERMINATING_DISTANCE)

### Configuration Constants

```python
COMPONENT_DISTANCE = 2      # Spacing between elements in array
TERMINATING_DISTANCE = 2    # Gap between output terminal pairs
```

---

## Unit Testing Guidance

### Testable Without Rendering

- `Array_Creator.draw_array()` → verify returned terminal positions relative to input
- MPPT anchor positions → verify pin coordinates match expected geometry
- Configuration loading → verify JSON parsed correctly into drawing parameters

### Suggested Test Structure

```
tests/
├── test_array_creator.py      # Terminal positions, element counts
├── test_mppt_component.py     # Anchor coordinates, segment count
├── test_config_loading.py     # JSON → drawing params mapping
└── test_drawing_generator.py  # End-to-end: config → SVG output exists
```

---

## Related Modules

| Module | Relationship |
|--------|-------------|
| `src/electrical_simulation` | Same circuit config → drawing is visual verification of simulated circuit |
| `src/power_cables` | Same `panels_per_string` determines array topology in both 2D and 3D |
| `constant/electrical/` | Shared source of truth for component specs and topology |

---

## Shared Parameterisation with Electrical Modules

The three electrical modules all derive topology from the same JSON:

```
constant/boat/{boat}.json
  └─ panels_per_string ──→ electrical_drawing: series count in Panel_Array
                        ──→ electrical_simulation: in_series for PySpice solar array
                        ──→ power_cables: string boundary for RED/BLUE wires

constant/electrical/{boat}_circuit_setup.json
  └─ mppt_panel.config_N.count ──→ electrical_drawing: number of MPPT blocks
                                ──→ electrical_simulation: number of MPPT instances
```

---

## Planned Work

- [ ] CLI with argparse (--circuit, --components, --boat-params, --output)
- [ ] Drawing generator that reads circuit_setup and builds full schematic
- [ ] Load component (motor block with power label)
- [ ] Battery component (with voltage/capacity label)
- [ ] Complete left-to-right layout: Panels → MPPT → Battery → Load
- [ ] Multi-MPPT stacking with shared bus connections
- [ ] SVG + PNG output to artifact directory
- [ ] Multi-boat support (rp1, rp2, rp3)
