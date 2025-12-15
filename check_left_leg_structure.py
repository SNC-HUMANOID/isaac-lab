#!/usr/bin/env python3
"""Check left leg structure in URDF for anomalies."""

import xml.etree.ElementTree as ET

urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.urdf"

tree = ET.parse(urdf_path)
root = tree.getroot()

print("=" * 80)
print("LEFT LEG STRUCTURE ANALYSIS")
print("=" * 80)

# Find all left leg joints
left_joints = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
               "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint"]

print("\n📊 LEFT LEG JOINTS:")
print("-" * 80)

for joint_name in left_joints:
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is not None:
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")

        xyz = origin.get("xyz") if origin is not None else "N/A"
        axis_xyz = axis.get("xyz") if axis is not None else "N/A"

        print(f"\n{joint_name}:")
        print(f"  Parent: {parent}")
        print(f"  Child:  {child}")
        print(f"  Origin XYZ: {xyz}")
        print(f"  Axis:       {axis_xyz}")

print("\n" + "=" * 80)
print("RIGHT LEG STRUCTURE (FOR COMPARISON)")
print("=" * 80)

# Find all right leg joints
right_joints = ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint"]

print("\n📊 RIGHT LEG JOINTS:")
print("-" * 80)

for joint_name in right_joints:
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is not None:
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")

        xyz = origin.get("xyz") if origin is not None else "N/A"
        axis_xyz = axis.get("xyz") if axis is not None else "N/A"

        print(f"\n{joint_name}:")
        print(f"  Parent: {parent}")
        print(f"  Child:  {child}")
        print(f"  Origin XYZ: {xyz}")
        print(f"  Axis:       {axis_xyz}")

print("\n" + "=" * 80)
print("COMPARISON ANALYSIS")
print("=" * 80)

# Compare axis directions
print("\n🔍 AXIS COMPARISON:")
for i, (left_j, right_j) in enumerate(zip(left_joints, right_joints)):
    left = root.find(f".//joint[@name='{left_j}']")
    right = root.find(f".//joint[@name='{right_j}']")

    if left is not None and right is not None:
        left_axis = left.find("axis").get("xyz")
        right_axis = right.find("axis").get("xyz")

        match = "✅ MATCH" if left_axis == right_axis or are_axes_symmetric(left_axis, right_axis) else "❌ DIFFERENT"

        print(f"\n{left_j.replace('left_', '')}:")
        print(f"  Left:  {left_axis}")
        print(f"  Right: {right_axis}")
        print(f"  {match}")

def are_axes_symmetric(left_str, right_str):
    """Check if axes are symmetric (e.g., X axis should be negated)."""
    left = [float(x) for x in left_str.split()]
    right = [float(x) for x in right_str.split()]

    # For symmetric robot, X component should be negated, Y and Z may vary
    return abs(left[0] + right[0]) < 0.01  # X should be opposite

print("\n" + "=" * 80)
