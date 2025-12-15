#!/usr/bin/env python3
"""Fix left hip axis in USD file directly using pxr.Usd library."""

import sys
sys.path.insert(0, "/home/sncbot/isaac-sim/extscache/omni.isaac.ml_archive-2.0.0+106.0.0.lx64.r.cp310/pip_prebundle")

from pxr import Usd, UsdPhysics

# Paths
usd_file = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.usd"

print(f"\n{'='*60}")
print(f"Fixing left hip axis in USD file")
print(f"{'='*60}")
print(f"USD file: {usd_file}\n")

# Open USD stage
stage = Usd.Stage.Open(usd_file)

if not stage:
    print(f"✗ Failed to open USD file")
    exit(1)

print("✓ USD file opened successfully")

# Find left hip pitch joint
left_hip_path = None
for prim in stage.Traverse():
    if "left_hip_pitch_joint" in str(prim.GetPath()):
        left_hip_path = prim.GetPath()
        print(f"✓ Found left hip joint at: {left_hip_path}")
        break

if not left_hip_path:
    print("✗ Could not find left_hip_pitch_joint in USD")
    exit(1)

# Get the joint prim
joint_prim = stage.GetPrimAtPath(left_hip_path)
if not joint_prim:
    print(f"✗ Could not get prim at {left_hip_path}")
    exit(1)

# Get the revolute joint
revolute_joint = UsdPhysics.RevoluteJoint(joint_prim)
if not revolute_joint:
    print(f"✗ Prim is not a RevoluteJoint")
    exit(1)

# Get current axis
current_axis = revolute_joint.GetAxisAttr().Get()
print(f"Current axis: {current_axis}")

# Fix axis: change Y from negative to positive
if current_axis:
    new_axis = (current_axis[0], abs(current_axis[1]), current_axis[2])
    print(f"New axis:     {new_axis}")

    # Set new axis
    revolute_joint.GetAxisAttr().Set(new_axis)
    print(f"✓ Axis updated")
else:
    print(f"✗ Could not get current axis")
    exit(1)

# Save USD
stage.Save()
print(f"✓ USD file saved")

print(f"\n{'='*60}")
print(f"✓ Left hip axis fixed successfully!")
print(f"  Old: Y = {current_axis[1]}")
print(f"  New: Y = {new_axis[1]}")
print(f"{'='*60}\n")
