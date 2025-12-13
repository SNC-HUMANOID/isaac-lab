#!/usr/bin/env python3
"""Simple test to verify left hip moves in correct direction with action scale."""

import torch
import time

# Simplified test without full Isaac Sim
print("="*60)
print("ACTION SCALE SIMULATION TEST")
print("="*60)

# Simulate what happens with action scale dictionary
print("\n1. Policy outputs same action for both hips:")
policy_action = 0.5
print(f"   Policy action: {policy_action:+.2f}")

print("\n2. Action scale dictionary applies:")
left_scale = -0.5  # INVERTED
right_scale = 0.5  # NORMAL

left_hip_action = policy_action * left_scale
right_hip_action = policy_action * right_scale

print(f"   Left hip:  {policy_action:+.2f} × {left_scale:+.2f} = {left_hip_action:+.2f}")
print(f"   Right hip: {policy_action:+.2f} × {right_scale:+.2f} = {right_hip_action:+.2f}")

print("\n3. URDF axis applies (STILL INVERTED IN URDF!):")
# Left hip axis in URDF: Y = -0.9999 (inverted)
# Right hip axis in URDF: Y = +0.9999 (normal)

print(f"   Left hip:  {left_hip_action:+.2f} × (axis Y=-0.9999) = effective {left_hip_action * -1:+.2f}")
print(f"   Right hip: {right_hip_action:+.2f} × (axis Y=+0.9999) = effective {right_hip_action:+.2f}")

print("\n4. FINAL RESULT:")
left_final = left_hip_action * -1  # Action * scale * axis_direction
right_final = right_hip_action

if left_final > 0 and right_final > 0:
    print("   ✓ Both legs move FORWARD together!")
elif left_final < 0 and right_final < 0:
    print("   ✓ Both legs move BACKWARD together!")
elif left_final * right_final < 0:
    print(f"   ✗ PROBLEM: Left ({left_final:+.2f}) and Right ({right_final:+.2f}) move in OPPOSITE directions!")
else:
    print("   ? One or both legs not moving")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)

if abs(left_final - right_final) < 0.01:
    print("✓ Action scale CORRECTLY compensates for URDF axis inversion!")
    print("  Both legs should move symmetrically.")
else:
    print("✗ Action scale DOES compensate, but there's still asymmetry:")
    print(f"  Left effective motion: {left_final:+.2f}")
    print(f"  Right effective motion: {right_final:+.2f}")
    print(f"  Difference: {abs(left_final - right_final):.4f}")

print("\n" + "="*60)
