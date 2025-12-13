#!/usr/bin/env python3

"""Show Humanoid_SNC robot in Isaac Sim using Isaac Lab"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Show Humanoid_SNC robot")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim
    app = AppLauncher(args)
    simulation_app = app.app

    # Import after app launch
    import omni.usd
    from pxr import Gf, UsdGeom, Usd
    from omni.isaac.core.utils.stage import get_current_stage

    print("\n" + "="*50)
    print("Humanoid_SNC Robot Viewer")
    print("="*50)

    # Get current stage
    stage = get_current_stage()
    
    # Load robot USD file
    robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
    robot_prim_path = "/World/Robot"
    
    try:
        # Load robot as reference
        robot_prim = stage.DefinePrim(robot_prim_path)
        robot_prim.GetReferences().AddReference(robot_usd_path)
        
        # Position robot above ground
        xformable = UsdGeom.Xformable(robot_prim)
        xform_op = xformable.AddTranslateOp()
        xform_op.Set(Gf.Vec3f(0.0, 0.0, 1.0))
        
        print(f"✓ Robot loaded: {robot_usd_path}")
        print(f"✓ Robot positioned at: (0, 0, 1)")
        
        # Add simple ground plane
        ground_prim_path = "/World/Ground"
        ground_prim = stage.DefinePrim(ground_prim_path, "Cube")
        cube_geom = UsdGeom.Cube(ground_prim)
        cube_geom.CreateSizeAttr(1.0)
        
        # Scale and position ground
        ground_xformable = UsdGeom.Xformable(ground_prim)
        scale_op = ground_xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(20.0, 20.0, 0.1))  # Large, thin ground
        translate_op = ground_xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3f(0.0, 0.0, -0.5))
        
        print("✓ Ground plane added")
        
    except Exception as e:
        print(f"✗ Error loading robot: {e}")
        simulation_app.close()
        return

    print("="*50)
    print("Robot successfully loaded!")
    print("Controls (if using GUI mode):")
    print("- Mouse: Orbit camera")
    print("- WASD: Move camera")
    print("- Q/E: Up/Down")
    print("- F: Focus on selection")
    print("- ESC: Exit")
    print("="*50)
    
    # Keep simulation running for a while in headless mode
    if args.headless:
        print("Running in headless mode for 5 seconds...")
        import time
        time.sleep(5)
        print("✓ Headless test completed successfully!")
    else:
        # Interactive mode - run until user closes
        print("Interactive mode - close window to exit...")
        while simulation_app.is_running():
            simulation_app.update()

    # Cleanup
    simulation_app.close()

if __name__ == "__main__":
    main()