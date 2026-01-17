#!/usr/bin/env python3
"""
Buoyancy equilibrium solver - finds the equilibrium pose of a boat.

Uses Newton-Raphson iteration to find the pose (z, pitch, roll) where:
1. Force equilibrium: buoyancy force = weight
2. Moment equilibrium: CoB is directly below CoG (no pitch/roll moments)

Usage:
    python -m src.buoyancy \
        --design artifact/boat.design.FCStd \
        --mass artifact/boat.mass.json \
        --materials constant/material/proa.json \
        --output artifact/boat.buoyancy.json
"""

import sys
import os
import json
import argparse
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import FreeCAD as App
except ImportError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    print("This script must be run with FreeCAD's Python", file=sys.stderr)
    sys.exit(1)

from src.physics.center_of_buoyancy import compute_center_of_buoyancy
from src.physics.center_of_mass import compute_center_of_gravity


def extract_hull_breakdown(cob_result: dict) -> dict:
    """
    Extract ama vs vaka breakdown from CoB result.

    Returns:
        Dictionary with:
        - ama_liters: total ama buoyancy volume
        - vaka_liters: total vaka buoyancy volume
        - ama_z_world: world z-coordinate of ama bottom reference point
        - vaka_z_world: world z-coordinate of vaka bottom reference point
    """
    ama_liters = 0.0
    vaka_liters = 0.0

    components = cob_result.get('components', [])

    for comp in components:
        pattern = comp.get('pattern', '')
        vol = comp.get('submerged_volume_liters', 0.0)

        if pattern.startswith('ama'):
            ama_liters += vol
        else:
            vaka_liters += vol

    # Get hull reference points in world frame
    hull_refs = cob_result.get('hull_refs', {})
    ama_z_world = hull_refs.get('ama_world', {}).get('z', 0.0)
    vaka_z_world = hull_refs.get('vaka_world', {}).get('z', 0.0)

    return {
        'ama_liters': round(ama_liters, 1),
        'vaka_liters': round(vaka_liters, 1),
        'ama_z_world': round(ama_z_world, 0),
        'vaka_z_world': round(vaka_z_world, 0)
    }


# Solver parameters
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_TOLERANCE = 1e-3  # Convergence tolerance for residuals
DEFAULT_Z_STEP = 10.0     # mm, for numerical Jacobian
DEFAULT_ANGLE_STEP = 0.1  # degrees, for numerical Jacobian


def transform_point(point: dict, z_displacement: float, pitch_deg: float,
                    roll_deg: float, rotation_center: dict) -> dict:
    """
    Transform a point by z displacement and pitch/roll rotations.

    The rotation is applied about rotation_center, then z displacement is added.
    """
    import math

    # Translate to rotation center
    x = point['x'] - rotation_center['x']
    y = point['y'] - rotation_center['y']
    z = point['z'] - rotation_center['z']

    # Apply rotation (same matrix as in center_of_buoyancy.py)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    cos_r = math.cos(roll_rad)
    sin_r = math.sin(roll_rad)

    # R = Ry(roll) * Rx(pitch)
    x_new = cos_r * x + sin_r * sin_p * y + sin_r * cos_p * z
    y_new = cos_p * y - sin_p * z
    z_new = -sin_r * x + cos_r * sin_p * y + cos_r * cos_p * z

    # Translate back and apply z displacement
    return {
        'x': x_new + rotation_center['x'],
        'y': y_new + rotation_center['y'],
        'z': z_new + rotation_center['z'] + z_displacement
    }


def compute_residuals(cog_result: dict, cob_result: dict,
                      z_displacement: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Compute equilibrium residuals.

    Both CoG and CoB must be in world coordinates for proper comparison.
    CoG is transformed from body frame using the current pose.

    Returns array of [force_residual, pitch_moment, roll_moment]:
    - force_residual: (buoyancy - weight) / weight  [normalized]
    - pitch_moment: (CoB_y - CoG_y) / 1000  [normalized to meters]
    - roll_moment: (CoB_x - CoG_x) / 1000  [normalized to meters]

    At equilibrium, all residuals should be zero.
    """
    weight_N = cog_result['weight_N']
    buoyancy_N = cob_result['buoyancy_force_N']

    # CoG in body frame (from mass calculation)
    CoG_body = cog_result['CoG']

    # Get rotation center from CoB result (same center used for hull transformation)
    rotation_center = cob_result['pose'].get('rotation_center', CoG_body)

    # Transform CoG to world frame using current pose
    CoG_world = transform_point(CoG_body, z_displacement, pitch_deg, roll_deg, rotation_center)

    # CoB is already in world frame (computed from transformed hull)
    CoB = cob_result['CoB']

    # Force residual (normalized by weight)
    force_residual = (buoyancy_N - weight_N) / weight_N if weight_N > 0 else 0

    # Moment residuals (normalized to meters)
    # For CoB to be directly below CoG, their x and y must match in world frame
    pitch_moment = (CoB['y'] - CoG_world['y']) / 1000.0  # Y offset causes pitch moment
    roll_moment = (CoB['x'] - CoG_world['x']) / 1000.0   # X offset causes roll moment

    return np.array([force_residual, pitch_moment, roll_moment])


def compute_jacobian(fcstd_path: str, cog_result: dict,
                     z: float, pitch: float, roll: float,
                     z_step: float = DEFAULT_Z_STEP,
                     angle_step: float = DEFAULT_ANGLE_STEP) -> np.ndarray:
    """
    Compute Jacobian matrix numerically using central differences.

    J[i,j] = d(residual_i) / d(state_j)

    State = [z, pitch, roll]
    Residuals = [force, pitch_moment, roll_moment]
    """
    J = np.zeros((3, 3))

    # Compute derivatives with respect to z
    cob_plus = compute_center_of_buoyancy(fcstd_path, z + z_step, pitch, roll)
    cob_minus = compute_center_of_buoyancy(fcstd_path, z - z_step, pitch, roll)
    r_plus = compute_residuals(cog_result, cob_plus, z + z_step, pitch, roll)
    r_minus = compute_residuals(cog_result, cob_minus, z - z_step, pitch, roll)
    J[:, 0] = (r_plus - r_minus) / (2 * z_step)

    # Compute derivatives with respect to pitch
    cob_plus = compute_center_of_buoyancy(fcstd_path, z, pitch + angle_step, roll)
    cob_minus = compute_center_of_buoyancy(fcstd_path, z, pitch - angle_step, roll)
    r_plus = compute_residuals(cog_result, cob_plus, z, pitch + angle_step, roll)
    r_minus = compute_residuals(cog_result, cob_minus, z, pitch - angle_step, roll)
    J[:, 1] = (r_plus - r_minus) / (2 * angle_step)

    # Compute derivatives with respect to roll
    cob_plus = compute_center_of_buoyancy(fcstd_path, z, pitch, roll + angle_step)
    cob_minus = compute_center_of_buoyancy(fcstd_path, z, pitch, roll - angle_step)
    r_plus = compute_residuals(cog_result, cob_plus, z, pitch, roll + angle_step)
    r_minus = compute_residuals(cog_result, cob_minus, z, pitch, roll - angle_step)
    J[:, 2] = (r_plus - r_minus) / (2 * angle_step)

    return J


def estimate_initial_z(cog_result: dict, fcstd_path: str) -> float:
    """
    Estimate initial z displacement to get buoyancy roughly equal to weight.

    Uses a simple search to find z where buoyancy is close to weight.
    """
    weight_N = cog_result['weight_N']

    # Binary search for z
    z_min, z_max = -5000.0, 0.0  # Search range in mm

    for _ in range(20):  # Binary search iterations
        z_mid = (z_min + z_max) / 2
        cob = compute_center_of_buoyancy(fcstd_path, z_mid, 0, 0)
        buoyancy_N = cob['buoyancy_force_N']

        if buoyancy_N < weight_N:
            z_max = z_mid  # Need to sink more (more negative z)
        else:
            z_min = z_mid  # Need to rise (less negative z)

        # Stop if close enough
        if abs(buoyancy_N - weight_N) / weight_N < 0.01:
            break

    return z_mid


def solve_equilibrium(fcstd_path: str, cog_result: dict,
                      max_iterations: int = DEFAULT_MAX_ITERATIONS,
                      tolerance: float = DEFAULT_TOLERANCE,
                      verbose: bool = True) -> dict:
    """
    Find equilibrium pose using Newton-Raphson iteration.

    Args:
        fcstd_path: Path to FreeCAD design file
        cog_result: Result from compute_center_of_gravity
        max_iterations: Maximum Newton-Raphson iterations
        tolerance: Convergence tolerance for residuals
        verbose: Print progress information

    Returns:
        Dictionary with equilibrium results
    """
    # Initial guess
    if verbose:
        print("  Estimating initial z displacement...")
    z = estimate_initial_z(cog_result, fcstd_path)
    pitch = 0.0
    roll = 0.0

    if verbose:
        print(f"  Initial guess: z={z:.1f}mm, pitch={pitch:.2f}°, roll={roll:.2f}°")

    iteration_history = []
    converged = False

    for iteration in range(max_iterations):
        # Compute CoB at current pose
        cob_result = compute_center_of_buoyancy(fcstd_path, z, pitch, roll)

        # Compute residuals (CoG is transformed to world frame internally)
        residuals = compute_residuals(cog_result, cob_result, z, pitch, roll)
        residual_norm = np.linalg.norm(residuals)

        # Extract hull breakdown for diagnostics
        hull_breakdown = extract_hull_breakdown(cob_result)

        # Record iteration
        iteration_history.append({
            'iteration': iteration,
            'z_mm': round(z, 2),
            'pitch_deg': round(pitch, 4),
            'roll_deg': round(roll, 4),
            'residual_norm': round(residual_norm, 6),
            'force_residual': round(residuals[0], 6),
            'pitch_residual': round(residuals[1], 6),
            'roll_residual': round(residuals[2], 6),
            'buoyancy_N': round(cob_result['buoyancy_force_N'], 2),
            'submerged_liters': round(cob_result['submerged_volume_liters'], 2),
            'ama_liters': hull_breakdown['ama_liters'],
            'vaka_liters': hull_breakdown['vaka_liters'],
            'ama_z_world': hull_breakdown['ama_z_world'],
            'vaka_z_world': hull_breakdown['vaka_z_world']
        })

        if verbose:
            print(f"  Iteration {iteration}: z={z:.1f}mm, pitch={pitch:.3f}°, "
                  f"roll={roll:.3f}°, |r|={residual_norm:.4f} "
                  f"[F:{residuals[0]:.3f}, P:{residuals[1]:.3f}, R:{residuals[2]:.3f}]")
            print(f"    Ama: {hull_breakdown['ama_liters']:.0f}L @ z={hull_breakdown['ama_z_world']:.0f}mm, "
                  f"Vaka: {hull_breakdown['vaka_liters']:.0f}L @ z={hull_breakdown['vaka_z_world']:.0f}mm")

        # Check convergence
        if residual_norm < tolerance:
            converged = True
            if verbose:
                print(f"  Converged after {iteration + 1} iterations")
            break

        # Compute Jacobian
        J = compute_jacobian(fcstd_path, cog_result, z, pitch, roll)

        # Check if Jacobian is singular
        det = np.linalg.det(J)
        if abs(det) < 1e-12:
            if verbose:
                print(f"  Warning: Jacobian nearly singular (det={det:.2e})")
            # Use pseudo-inverse
            try:
                delta = np.linalg.lstsq(J, -residuals, rcond=None)[0]
            except np.linalg.LinAlgError:
                if verbose:
                    print("  Error: Could not solve linear system")
                break
        else:
            # Newton-Raphson update
            delta = np.linalg.solve(J, -residuals)

        # Limit maximum step size for stability
        max_z_step = 200.0  # mm
        max_angle_step = 5.0  # degrees

        step_scale = 1.0
        if abs(delta[0]) > max_z_step:
            step_scale = min(step_scale, max_z_step / abs(delta[0]))
        if abs(delta[1]) > max_angle_step:
            step_scale = min(step_scale, max_angle_step / abs(delta[1]))
        if abs(delta[2]) > max_angle_step:
            step_scale = min(step_scale, max_angle_step / abs(delta[2]))

        if step_scale < 1.0:
            delta = delta * step_scale
            if verbose:
                print(f"    Step scaled by {step_scale:.3f} for stability")

        # Backtracking line search (Armijo condition)
        # Find step size alpha such that residual decreases
        alpha = 1.0
        min_alpha = 0.01
        backtrack_factor = 0.5

        # Track best step that actually improves residual
        best_alpha = 0.0  # No step by default
        best_norm = residual_norm  # Must beat current residual

        while alpha >= min_alpha:
            # Try step with current alpha
            z_new = z + alpha * delta[0]
            pitch_new = pitch + alpha * delta[1]
            roll_new = roll + alpha * delta[2]

            # Evaluate residual at new point
            cob_new = compute_center_of_buoyancy(fcstd_path, z_new, pitch_new, roll_new)
            residuals_new = compute_residuals(cog_result, cob_new, z_new, pitch_new, roll_new)
            new_norm = np.linalg.norm(residuals_new)

            # Track best improvement
            if new_norm < best_norm:
                best_alpha = alpha
                best_norm = new_norm

            # Armijo condition: sufficient decrease
            if new_norm < residual_norm * (1 - 0.1 * alpha):
                break

            alpha *= backtrack_factor

        # Use best alpha found
        if alpha < min_alpha:
            alpha = best_alpha
            if alpha == 0.0:
                if verbose:
                    print(f"    Line search failed: no improvement found, taking small step")
                alpha = min_alpha  # Take minimal step to avoid getting stuck
            else:
                if verbose:
                    print(f"    Line search: using best α={alpha:.3f}")

        # Update state with chosen step size
        z += alpha * delta[0]
        pitch += alpha * delta[1]
        roll += alpha * delta[2]

    # Final CoB computation
    final_cob = compute_center_of_buoyancy(fcstd_path, z, pitch, roll)
    final_residuals = compute_residuals(cog_result, final_cob, z, pitch, roll)

    # Transform CoG to world frame for output
    rotation_center = final_cob['pose'].get('rotation_center', cog_result['CoG'])
    cog_world = transform_point(cog_result['CoG'], z, pitch, roll, rotation_center)

    # Extract ama/vaka breakdown
    hull_breakdown = extract_hull_breakdown(final_cob)

    return {
        'converged': converged,
        'iterations': len(iteration_history),
        'equilibrium': {
            'z_offset_mm': round(z, 2),
            'pitch_deg': round(pitch, 4),
            'roll_deg': round(roll, 4)
        },
        'center_of_gravity_body': cog_result['CoG'],
        'center_of_gravity_world': {
            'x': round(cog_world['x'], 2),
            'y': round(cog_world['y'], 2),
            'z': round(cog_world['z'], 2)
        },
        'center_of_buoyancy': final_cob['CoB'],
        'total_mass_kg': round(cog_result['total_mass_kg'], 2),
        'weight_N': round(cog_result['weight_N'], 2),
        'buoyancy_force_N': round(final_cob['buoyancy_force_N'], 2),
        'submerged_volume_liters': round(final_cob['submerged_volume_liters'], 2),
        'ama': {
            'submerged_volume_liters': hull_breakdown['ama_liters'],
            'z_world_mm': hull_breakdown['ama_z_world']
        },
        'vaka': {
            'submerged_volume_liters': hull_breakdown['vaka_liters'],
            'z_world_mm': hull_breakdown['vaka_z_world']
        },
        'final_residuals': {
            'force': round(final_residuals[0], 6),
            'pitch_moment': round(final_residuals[1], 6),
            'roll_moment': round(final_residuals[2], 6),
            'norm': round(np.linalg.norm(final_residuals), 6)
        },
        'iteration_history': iteration_history
    }


def main():
    parser = argparse.ArgumentParser(
        description='Find buoyancy equilibrium pose using Newton-Raphson',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--design', required=True,
                        help='Path to FCStd design file')
    parser.add_argument('--mass', required=False,
                        help='Path to mass.json artifact (optional, faster)')
    parser.add_argument('--materials', required=True,
                        help='Path to materials JSON file')
    parser.add_argument('--output', required=True,
                        help='Path to output JSON file')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f'Maximum iterations (default: {DEFAULT_MAX_ITERATIONS})')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE,
                        help=f'Convergence tolerance (default: {DEFAULT_TOLERANCE})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    if not os.path.exists(args.design):
        print(f"ERROR: Design file not found: {args.design}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    if verbose:
        print(f"Solving buoyancy equilibrium: {args.design}")

    # Compute center of gravity
    if verbose:
        print("  Computing center of gravity...")

    if args.mass and os.path.exists(args.mass):
        from src.physics.center_of_mass import compute_cog_from_mass_artifact
        cog_result = compute_cog_from_mass_artifact(args.mass, args.design)
    else:
        cog_result = compute_center_of_gravity(args.design, args.materials)

    if verbose:
        print(f"  CoG: ({cog_result['CoG']['x']:.1f}, {cog_result['CoG']['y']:.1f}, "
              f"{cog_result['CoG']['z']:.1f}) mm")
        print(f"  Total mass: {cog_result['total_mass_kg']:.2f} kg")
        print(f"  Weight: {cog_result['weight_N']:.2f} N")

    # Solve equilibrium
    if verbose:
        print("  Running Newton-Raphson solver...")

    result = solve_equilibrium(
        args.design,
        cog_result,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        verbose=verbose
    )

    # Add validator field
    result['validator'] = 'buoyancy'

    # Write output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"✓ Buoyancy equilibrium {'found' if result['converged'] else 'NOT CONVERGED'}")
        eq = result['equilibrium']
        print(f"  Equilibrium pose:")
        print(f"    z offset: {eq['z_offset_mm']:.2f} mm")
        print(f"    pitch: {eq['pitch_deg']:.4f}°")
        print(f"    roll: {eq['roll_deg']:.4f}°")
        cog_w = result['center_of_gravity_world']
        cob = result['center_of_buoyancy']
        print(f"  CoG (world): ({cog_w['x']:.1f}, {cog_w['y']:.1f}, {cog_w['z']:.1f}) mm")
        print(f"  CoB (world): ({cob['x']:.1f}, {cob['y']:.1f}, {cob['z']:.1f}) mm")
        print(f"  Submerged volume: {result['submerged_volume_liters']:.2f} liters")
        ama = result['ama']
        vaka = result['vaka']
        print(f"    Ama: {ama['submerged_volume_liters']:.1f}L @ z={ama['z_world_mm']:.0f}mm")
        print(f"    Vaka: {vaka['submerged_volume_liters']:.1f}L @ z={vaka['z_world_mm']:.0f}mm")
        print(f"  Buoyancy force: {result['buoyancy_force_N']:.2f} N")
        print(f"  Output: {args.output}")

    # Exit with error if not converged
    if not result['converged']:
        sys.exit(1)


if __name__ == "__main__":
    main()
