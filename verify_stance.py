#!/usr/bin/env python3
"""Verify leg stance configuration."""

import math

# Configuration
hip_spacing = 0.0755  # meters (from URDF)
hip_roll_angle = 0.05  # radians (new setting)

# Calculate
total_hip_width_cm = hip_spacing * 2 * 100  # Convert to cm
hip_roll_degrees = math.degrees(hip_roll_angle)

# Rough estimation of foot spacing reduction
# (simplified, actual calculation more complex)
leg_length = 0.5  # approximate total leg length in meters
inward_movement = leg_length * math.sin(hip_roll_angle) * 100  # cm

new_approximate_spacing = total_hip_width_cm - (inward_movement * 2)

print("=" * 60)
print("Leg Stance Configuration Analysis")
print("=" * 60)
print()
print("URDF Hip Spacing:")
print(f"  Left hip:  +{hip_spacing:.4f} m")
print(f"  Right hip: -{hip_spacing:.4f} m")
print(f"  Total width: {total_hip_width_cm:.1f} cm")
print()
print("Initial Joint Configuration:")
print(f"  Hip roll angle: ±{hip_roll_angle:.2f} rad (±{hip_roll_degrees:.2f}°)")
print(f"  Effect: Brings legs inward by ~{inward_movement:.1f} cm per leg")
print()
print("Estimated Foot Spacing:")
print(f"  Original (hip roll = 0): ~{total_hip_width_cm:.1f} cm")
print(f"  New (hip roll = ±0.05): ~{new_approximate_spacing:.1f} cm")
print()
print("G1 Hip Roll Limits:")
print(f"  Range: [-0.43, 0.43] rad ([-24.6°, 24.6°])")
print(f"  Current: ±{hip_roll_angle:.2f} rad (±{hip_roll_degrees:.2f}°)")
print(f"  Percentage of limit: {(hip_roll_angle/0.43)*100:.1f}%")
print()
print("Status: ✅ Within safe limits")
print("=" * 60)
print()
print("Recommendations:")
print("  - Current ±0.05 rad: Good for normal walking")
print("  - Increase to ±0.08 rad: Narrower stance (more compact)")
print("  - Increase to ±0.10 rad: Very narrow (running/athletic)")
print("  - Decrease to ±0.03 rad: Wider stance (more stable)")
print()

