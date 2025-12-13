#!/usr/bin/env python3
"""Verify full left-right leg symmetry in URDF."""

import xml.etree.ElementTree as ET

urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.urdf"
tree = ET.parse(urdf_path)
root = tree.getroot()

print("="*80)
print("FULL LEG SYMMETRY VERIFICATION")
print("="*80)

leg_joints = [
    "hip_pitch_joint",
    "hip_roll_joint",
    "hip_yaw_joint",
    "knee_joint",
    "ankle_roll_joint",
    "ankle_pitch_joint"
]

all_symmetric = True

for joint_name in leg_joints:
    left = root.find(f".//joint[@name='left_{joint_name}']")
    right = root.find(f".//joint[@name='right_{joint_name}']")

    if left and right:
        left_origin = left.find('origin').get('xyz')
        right_origin = right.find('origin').get('xyz')

        left_vals = [float(x) for x in left_origin.split()]
        right_vals = [float(x) for x in right_origin.split()]

        # Check symmetry: X same, Y negated, Z same
        x_match = abs(left_vals[0] - right_vals[0]) < 0.001
        y_mirror = abs(left_vals[1] + right_vals[1]) < 0.001
        z_match = abs(left_vals[2] - right_vals[2]) < 0.001

        is_symmetric = x_match and y_mirror and z_match
        status = "✅" if is_symmetric else "❌"

        print(f"\n{status} {joint_name}:")
        print(f"   Left:  {left_origin}")
        print(f"   Right: {right_origin}")

        if not is_symmetric:
            print(f"   ⚠️  X match: {x_match}, Y mirror: {y_mirror}, Z match: {z_match}")
            all_symmetric = False

print("\n" + "="*80)
if all_symmetric:
    print("✅ ALL LEG JOINTS ARE SYMMETRIC!")
else:
    print("❌ SOME JOINTS ARE NOT SYMMETRIC - CHECK ABOVE")
print("="*80)
