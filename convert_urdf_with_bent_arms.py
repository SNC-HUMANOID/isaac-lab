#!/usr/bin/env python3
"""Convert URDF to USD with bent arm pose (90 degrees)."""

import os
from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": True})

import omni
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.importer.urdf")

from omni.importer.urdf import _urdf
import omni.isaac.core.utils.stage as stage_utils
from pxr import Gf, UsdPhysics

# Paths
urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.urdf"
output_usd_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_bent_arms.usd"

print("="*80)
print("CONVERTING URDF TO USD WITH BENT ARMS")
print("="*80)
print(f"Input URDF: {urdf_path}")
print(f"Output USD: {output_usd_path}")

# Import URDF
urdf_interface = _urdf.acquire_urdf_interface()

# Import settings
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.make_default_prim = True
import_config.self_collision = False
import_config.create_physics_scene = True
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
import_config.default_position_drive_damping = 10.0
import_config.default_position_drive_stiffness = 200.0

# Convert
print("\n🔄 Converting URDF to USD...")
success, prim_path = urdf_interface.parse_urdf(urdf_path, import_config)

if not success:
    print("❌ Failed to import URDF")
    simulation_app.close()
    exit(1)

print(f"✅ URDF imported to prim: {prim_path}")

# Get stage
stage = omni.usd.get_context().get_stage()
robot_prim = stage.GetPrimAtPath(prim_path)

print("\n🔧 Setting arm joint positions to bent (90 degrees)...")

# Set elbow joints to -90 degrees (-1.57 radians)
elbow_joints = [
    "left_elbow_1_joint",
    "left_elbow_2_joint",
    "right_elbow_1_joint",
    "right_elbow_2_joint"
]

for joint_name in elbow_joints:
    joint_path = f"{prim_path}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_path)

    if joint_prim.IsValid():
        # Get revolute joint
        from pxr import UsdPhysics
        joint = UsdPhysics.RevoluteJoint(joint_prim)

        # Set default position to -90 degrees
        drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if drive_api:
            drive_api.GetTargetPositionAttr().Set(-1.57)  # -90 degrees
            print(f"  ✅ {joint_name}: set to -90° (-1.57 rad)")
        else:
            print(f"  ⚠️  {joint_name}: drive API not found")
    else:
        print(f"  ❌ {joint_name}: joint not found")

# Save USD
print(f"\n💾 Saving USD to: {output_usd_path}")
stage.Export(output_usd_path)

print("\n" + "="*80)
print("✅ CONVERSION COMPLETE!")
print("="*80)
print(f"\nYou can now import: {output_usd_path}")
print("The arms should be bent at 90 degrees")

simulation_app.close()
