#!/usr/bin/env python3
"""Fix hip roll joint symmetry in URDF."""

import xml.etree.ElementTree as ET
import shutil
from datetime import datetime

urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.urdf"

# Backup
backup_path = urdf_path + f".backup_hip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(urdf_path, backup_path)
print(f"✅ Backup created: {backup_path}")

# Load URDF
tree = ET.parse(urdf_path)
root = tree.getroot()

print("\n" + "="*80)
print("FIXING HIP ROLL JOINT SYMMETRY")
print("="*80)

# Get joints
left_hip_roll = root.find(".//joint[@name='left_hip_roll_joint']")
right_hip_roll = root.find(".//joint[@name='right_hip_roll_joint']")

# Print before
print("\n📊 BEFORE FIX:")
left_before = left_hip_roll.find('origin').get('xyz')
right_before = right_hip_roll.find('origin').get('xyz')
print(f"Left hip roll origin:  {left_before}")
print(f"Right hip roll origin: {right_before}")

# Fix: Make left mirror right
right_origin = right_hip_roll.find('origin').get('xyz')
right_vals = [float(x) for x in right_origin.split()]

# For left hip roll: same X, negated Y, same Z
left_vals = [right_vals[0], -right_vals[1], right_vals[2]]
left_origin_new = f"{left_vals[0]} {left_vals[1]} {left_vals[2]}"

left_hip_roll.find('origin').set('xyz', left_origin_new)

# Print after
print("\n📊 AFTER FIX:")
left_after = left_hip_roll.find('origin').get('xyz')
right_after = right_hip_roll.find('origin').get('xyz')
print(f"Left hip roll origin:  {left_after}")
print(f"Right hip roll origin: {right_after}")

# Verification
left_vals_new = [float(x) for x in left_after.split()]
right_vals_new = [float(x) for x in right_after.split()]

print("\n✅ VERIFICATION:")
print(f"  X values: left={left_vals_new[0]:.6f}, right={right_vals_new[0]:.6f} (should be same)")
print(f"  Y values: left={left_vals_new[1]:.6f}, right={right_vals_new[1]:.6f} (should be negated)")
print(f"  Z values: left={left_vals_new[2]:.6f}, right={right_vals_new[2]:.6f} (should be same)")

x_match = abs(left_vals_new[0] - right_vals_new[0]) < 0.0001
y_mirror = abs(left_vals_new[1] + right_vals_new[1]) < 0.0001
z_match = abs(left_vals_new[2] - right_vals_new[2]) < 0.0001

if x_match and y_mirror and z_match:
    print("\n✅ HIP ROLL JOINTS ARE NOW SYMMETRIC!")
else:
    print(f"\n⚠️  Warning: X match={x_match}, Y mirror={y_mirror}, Z match={z_match}")

# Save
tree.write(urdf_path, encoding='utf-8', xml_declaration=True)
print(f"\n✅ Fixed URDF saved to: {urdf_path}")
print("="*80)
