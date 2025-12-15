#!/usr/bin/env python3
"""Simple URDF to USD conversion using Isaac Sim asset converter."""

import os
from pxr import Usd, UsdGeom
from isaacsim import SimulationApp

# Start Isaac Sim headless
simulation_app = SimulationApp({"headless": True})

import omni
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.urdf")

import omni.kit.commands
from omni.isaac.urdf import _urdf

# Paths
urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_hip_axis.urdf"
usd_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_hip_axis.usd"

print(f"\n{'='*60}")
print(f"URDF to USD Conversion")
print(f"{'='*60}")
print(f"Input URDF:  {urdf_path}")
print(f"Output USD:  {usd_path}")
print(f"{'='*60}\n")

# Import URDF
status, import_config = omni.kit.commands.execute(
    "URDFCreateImportConfig"
)

import_config.merge_fixed_joints = False
import_config.fix_base = False
import_config.make_default_prim = True
import_config.create_physics_scene = False
import_config.import_inertia_tensor = True
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
import_config.default_position_drive_damping = 1000.0
import_config.default_position_drive_stiffness = 10000.0

print("Importing URDF...")
status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
)

if not status:
    print(f"✗ Failed to import URDF")
    simulation_app.close()
    exit(1)

print(f"✓ URDF imported to stage at: {prim_path}")

# Save as USD
print(f"\nSaving to USD file...")
stage = omni.usd.get_context().get_stage()

# Remove old file if exists
if os.path.exists(usd_path):
    os.remove(usd_path)
    print(f"  Removed old USD file")

stage.Export(usd_path)
print(f"✓ USD file saved: {usd_path}")

print(f"\n{'='*60}")
print(f"✓ Conversion complete!")
print(f"{'='*60}\n")

simulation_app.close()
