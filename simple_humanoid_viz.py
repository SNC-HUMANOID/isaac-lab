#!/usr/bin/env python3

"""Simple script to visualize Humanoid_SNC robot in Isaac Sim"""

import argparse
from isaaclab.app import AppLauncher

def main():
    # Parse arguments  
    parser = argparse.ArgumentParser(description="Simple Humanoid_SNC robot visualization")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim
    app = AppLauncher(args)
    simulation_app = app.app

    # Import Isaac Sim modules after app launch
    import omni.usd
    from pxr import Gf, UsdGeom
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage

    print("\n" + "="*60)
    print("Simple Humanoid_SNC Visualization")
    print("="*60)
    print("Loading robot model...")
    
    # Create world
    world = World(stage_units_in_meters=1.0)
    
    # Add ground plane (optional - simple cube as ground)
    ground = DynamicCuboid(
        prim_path="/World/Ground",
        name="ground",
        position=(0, 0, -0.5),
        size=(10.0, 10.0, 1.0),
        color=(0.5, 0.5, 0.5)
    )
    
    # Add robot USD file to stage
    robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
    robot_prim_path = "/World/Humanoid_SNC"
    
    try:
        add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
        print(f"✓ Successfully loaded robot from: {robot_usd_path}")
        
        # Set robot position
        robot_prim = UsdGeom.Xformable.Get(world.stage, robot_prim_path)
        if robot_prim:
            xform_op = robot_prim.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
            xform_op.Set(Gf.Vec3d(0.0, 0.0, 1.0))  # 1m above ground
            print("✓ Robot positioned at (0, 0, 1)")
            
    except Exception as e:
        print(f"✗ Failed to load robot: {e}")
        return

    # Reset world to apply changes
    world.reset()
    
    print("="*60)
    print("Controls:")
    print("- Mouse: Orbit camera around robot")  
    print("- WASD: Move camera position")
    print("- Q/E: Move camera up/down")
    print("- F: Focus on robot")
    print("- Press ESC or close window to exit")
    print("="*60)
    
    # Simulation loop
    while simulation_app.is_running():
        world.step(render=True)
        
    # Cleanup
    simulation_app.close()
    print("Visualization closed.")

if __name__ == "__main__":
    main()