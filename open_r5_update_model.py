#!/usr/bin/env python3
"""Open SNC R5 UPDATE model in Isaac Sim."""

from isaacsim import SimulationApp

# Launch Isaac Sim with GUI
simulation_app = SimulationApp({"headless": False})

import omni
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.importer.urdf")

from omni.importer.urdf import _urdf
import carb

# URDF path
urdf_path = "/home/sncbot/Downloads/SNC_R5_UPDATE/Humanoid_SNC.urdf"

print("="*80)
print("🤖 SNC R5 UPDATE MODEL VIEWER")
print("="*80)
print(f"📁 URDF: {urdf_path}")
print(f"📁 STL Directory: /home/sncbot/Downloads/SNC_R5_UPDATE/")
print("\n🔄 Importing URDF...")

# Get URDF importer interface
urdf_interface = _urdf.acquire_urdf_interface()

# Import configuration
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

# Parse URDF
success, prim_path = urdf_interface.parse_urdf(urdf_path, import_config)

if success:
    print(f"\n✅ Robot imported successfully!")
    print(f"   Prim path: {prim_path}")
    print("\n💡 Controls:")
    print("   - Mouse: Rotate camera")
    print("   - Scroll: Zoom")
    print("   - WASD: Move camera")
    print("   - SPACE: Play/Pause simulation")
    print("   - ESC: Stop")
    print("\n📊 Check Property panel to see:")
    print("   - Joints and their positions")
    print("   - Body masses and inertias")
    print("   - Collision shapes")
    print("="*80)
else:
    print("❌ Failed to import URDF")
    carb.log_error("URDF import failed")

# Keep simulation running
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
