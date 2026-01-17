#!/bin/bash
# Test utilities for the physics library
# These are development tools, not pipeline stages.
#
# Usage:
#   ./scripts/test_physics.sh cog       # Test center of gravity
#   ./scripts/test_physics.sh cob       # Test center of buoyancy
#   ./scripts/test_physics.sh buoyancy  # Test buoyancy equilibrium solver
#   ./scripts/test_physics.sh all       # Run all tests

set -e

# Configuration - adjust these for your test case
BOAT="rp2"
CONFIG="beaching"
DESIGN="artifact/${BOAT}.${CONFIG}.design.FCStd"
MATERIALS="constant/material/proa.json"
MASS_ARTIFACT="artifact/${BOAT}.${CONFIG}.mass.json"

# Output to tmp to avoid polluting artifact/
OUTPUT_DIR="/tmp/physics-test"
mkdir -p "$OUTPUT_DIR"

# Detect platform and set FreeCAD Python path
if [[ "$OSTYPE" == "darwin"* ]]; then
    FREECAD_BUNDLE="/Applications/FreeCAD.app"
    FREECAD_PYTHON="$FREECAD_BUNDLE/Contents/Resources/bin/python"
    export PYTHONPATH="$FREECAD_BUNDLE/Contents/Resources/lib:$FREECAD_BUNDLE/Contents/Resources/Mod:$PWD"
    export DYLD_LIBRARY_PATH="$FREECAD_BUNDLE/Contents/Frameworks:$FREECAD_BUNDLE/Contents/Resources/lib"
else
    FREECAD_PYTHON="python3"
    export PYTHONPATH="$PWD"
fi

test_cog() {
    echo "=== Testing Center of Gravity ==="
    echo ""

    # From geometry + materials
    echo "Method 1: From geometry + materials"
    $FREECAD_PYTHON -m src.physics cog \
        --design "$DESIGN" \
        --materials "$MATERIALS" \
        --output "$OUTPUT_DIR/cog.json"
    echo ""

    # From mass artifact (if it exists)
    if [[ -f "$MASS_ARTIFACT" ]]; then
        echo "Method 2: From mass artifact (faster)"
        $FREECAD_PYTHON -m src.physics cog \
            --design "$DESIGN" \
            --mass-artifact "$MASS_ARTIFACT" \
            --output "$OUTPUT_DIR/cog_from_mass.json"
        echo ""
    fi

    echo "Output: $OUTPUT_DIR/cog.json"
    echo ""
}

test_cob() {
    echo "=== Testing Center of Buoyancy ==="
    echo ""

    # At waterline (z=0)
    echo "Pose 1: At waterline (z=0, no rotation)"
    $FREECAD_PYTHON -m src.physics cob \
        --design "$DESIGN" \
        --z 0 --pitch 0 --roll 0 \
        --output "$OUTPUT_DIR/cob_z0.json"
    echo ""

    # Sunk 100mm
    echo "Pose 2: Sunk 100mm"
    $FREECAD_PYTHON -m src.physics cob \
        --design "$DESIGN" \
        --z -500 --pitch 0 --roll 0 \
        --output "$OUTPUT_DIR/cob_z-100.json"
    echo ""

    # Sunk with pitch
    echo "Pose 3: Sunk 100mm with 2° pitch"
    $FREECAD_PYTHON -m src.physics cob \
        --design "$DESIGN" \
        --z -500 --pitch 2.0 --roll 0 \
        --output "$OUTPUT_DIR/cob_pitched.json"
    echo ""

    echo "Outputs in: $OUTPUT_DIR/"
    echo ""
}

test_buoyancy() {
    echo "=== Testing Buoyancy Equilibrium Solver ==="
    echo ""

    # Check if mass artifact exists
    if [[ ! -f "$MASS_ARTIFACT" ]]; then
        echo "Mass artifact not found: $MASS_ARTIFACT"
        echo "Run 'make mass BOAT=$BOAT CONFIGURATION=$CONFIG' first"
        exit 1
    fi

    echo "Running Newton-Raphson equilibrium solver..."
    $FREECAD_PYTHON -m src.buoyancy \
        --design "$DESIGN" \
        --mass "$MASS_ARTIFACT" \
        --materials "$MATERIALS" \
        --output "$OUTPUT_DIR/buoyancy.json"
    echo ""

    echo "Output: $OUTPUT_DIR/buoyancy.json"
    echo ""
}

case "${1:-all}" in
    cog)
        test_cog
        ;;
    cob)
        test_cob
        ;;
    buoyancy)
        test_buoyancy
        ;;
    all)
        test_cog
        test_cob
        test_buoyancy
        ;;
    *)
        echo "Usage: $0 {cog|cob|buoyancy|all}"
        exit 1
        ;;
esac

echo "=== Done ==="
