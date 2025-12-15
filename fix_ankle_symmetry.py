#!/usr/bin/env python3
"""Fix ankle joint symmetry in URDF by making left ankle mirror right ankle."""

import xml.etree.ElementTree as ET
import shutil
from datetime import datetime

urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.urdf"

# Backup
backup_path = urdf_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(urdf_path, backup_path)
print(f"✅ Backup created: {backup_path}")

# Load URDF
tree = ET.parse(urdf_path)
root = tree.getroot()

print("\n" + "="*80)
print("FIXING ANKLE JOINT SYMMETRY")
print("="*80)

# Get right ankle joints (we'll use these as reference)
right_ankle_roll = root.find(".//joint[@name='right_ankle_roll_joint']")
right_ankle_pitch = root.find(".//joint[@name='right_ankle_pitch_joint']")

# Get left ankle joints (we'll fix these)
left_ankle_roll = root.find(".//joint[@name='left_ankle_roll_joint']")
left_ankle_pitch = root.find(".//joint[@name='left_ankle_pitch_joint']")

def mirror_xyz(xyz_str, mirror_y=True, mirror_x=False):
    """Mirror XYZ coordinates for symmetric robot."""
    coords = [float(x) for x in xyz_str.split()]
    if mirror_x:
        coords[0] = -coords[0]
    if mirror_y:
        coords[1] = -coords[1]
    return f"{coords[0]} {coords[1]} {coords[2]}"

def mirror_axis(axis_str, mirror_x=True):
    """Mirror axis direction for symmetric robot."""
    coords = [float(x) for x in axis_str.split()]
    # For left leg, X component should typically be same or negated depending on joint type
    # Y component should be negated
    # Z component should be same
    # But for ankle roll/pitch, we keep same direction
    return axis_str  # Keep axis same for ankles

# Print before
print("\n📊 BEFORE FIX:")
print(f"Left ankle roll origin:  {left_ankle_roll.find('origin').get('xyz')}")
print(f"Right ankle roll origin: {right_ankle_roll.find('origin').get('xyz')}")
print(f"Left ankle pitch origin:  {left_ankle_pitch.find('origin').get('xyz')}")
print(f"Right ankle pitch origin: {right_ankle_pitch.find('origin').get('xyz')}")

# Fix left ankle roll to mirror right ankle roll
right_roll_origin = right_ankle_roll.find('origin').get('xyz')
right_roll_axis = right_ankle_roll.find('axis').get('xyz')

# For symmetric humanoid: left ankle should have same X, negated Y, same Z
left_roll_origin_xyz = mirror_xyz(right_roll_origin, mirror_y=True, mirror_x=False)
left_ankle_roll.find('origin').set('xyz', left_roll_origin_xyz)
left_ankle_roll.find('axis').set('xyz', right_roll_axis)  # Use same axis

# Fix left ankle pitch to mirror right ankle pitch
right_pitch_origin = right_ankle_pitch.find('origin').get('xyz')
right_pitch_axis = right_ankle_pitch.find('axis').get('xyz')

left_pitch_origin_xyz = mirror_xyz(right_pitch_origin, mirror_y=True, mirror_x=False)
left_ankle_pitch.find('origin').set('xyz', left_pitch_origin_xyz)
left_ankle_pitch.find('axis').set('xyz', right_pitch_axis)  # Use same axis

# Print after
print("\n📊 AFTER FIX:")
print(f"Left ankle roll origin:  {left_ankle_roll.find('origin').get('xyz')}")
print(f"Right ankle roll origin: {right_ankle_roll.find('origin').get('xyz')}")
print(f"Left ankle pitch origin:  {left_ankle_pitch.find('origin').get('xyz')}")
print(f"Right ankle pitch origin: {right_ankle_pitch.find('origin').get('xyz')}")

print("\n✅ VERIFICATION:")
left_vals = [float(x) for x in left_ankle_roll.find('origin').get('xyz').split()]
right_vals = [float(x) for x in right_ankle_roll.find('origin').get('xyz').split()]
print(f"  X values: left={left_vals[0]:.4f}, right={right_vals[0]:.4f} (should be same)")
print(f"  Y values: left={left_vals[1]:.4f}, right={right_vals[1]:.4f} (should be negated)")
print(f"  Z values: left={left_vals[2]:.4f}, right={right_vals[2]:.4f} (should be similar)")

# Save modified URDF
tree.write(urdf_path, encoding='utf-8', xml_declaration=True)
print(f"\n✅ Fixed URDF saved to: {urdf_path}")

print("\n" + "="*80)
print("✅ ANKLE SYMMETRY FIX COMPLETE!")
print("="*80)
