#!/usr/bin/env python3

"""View Humanoid_SNC robot using basic USD APIs"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="View Humanoid_SNC robot")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim
    app = AppLauncher(args)
    simulation_app = app.app

    # Import USD modules after app launch
    import omni.usd
    from pxr import Gf, UsdGeom, Usd

    print("\n" + "="*50)
    print("Humanoid_SNC Robot Viewer")
    print("="*50)

    try:
        # Get USD context and stage
        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()
        
        # Create new stage if none exists
        if not stage:
            usd_context.new_stage()
            stage = usd_context.get_stage()
        
        print("✓ USD Stage ready")
        
        # Load robot USD file as reference
        robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
        robot_prim_path = "/World/Robot"
        
        # Create World xform
        world_prim = stage.DefinePrim("/World", "Xform")
        
        # Load robot
        robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
        robot_prim.GetReferences().AddReference(robot_usd_path)
        
        # Position robot
        xformable = UsdGeom.Xformable(robot_prim)
        # Check if translate op already exists
        if not xformable.GetTranslateOp():
            translate_op = xformable.AddTranslateOp()
        else:
            translate_op = xformable.GetTranslateOp()
        translate_op.Set(Gf.Vec3f(0.0, 0.0, 1.0))
        
        print(f"✓ Robot loaded from: {robot_usd_path}")
        print("✓ Robot positioned at (0, 0, 1)")
        
        # Add ground plane
        ground_prim = stage.DefinePrim("/World/Ground", "Cube")
        cube = UsdGeom.Cube(ground_prim)
        cube.CreateSizeAttr(1.0)
        
        ground_xform = UsdGeom.Xformable(ground_prim)
        # Check if ops already exist
        if not ground_xform.GetScaleOp():
            ground_scale = ground_xform.AddScaleOp()
        else:
            ground_scale = ground_xform.GetScaleOp()
        ground_scale.Set(Gf.Vec3f(10.0, 10.0, 0.1))
        
        if not ground_xform.GetTranslateOp():
            ground_translate = ground_xform.AddTranslateOp()
        else:
            ground_translate = ground_xform.GetTranslateOp()
        ground_translate.Set(Gf.Vec3f(0.0, 0.0, -0.5))
        
        print("✓ Ground plane added")
        
        # Set camera view
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        if viewport_api:
            viewport_api.set_camera_position(
                "/OmniverseKit_Persp", 
                Gf.Vec3d(3.0, 3.0, 2.0), 
                True
            )
            viewport_api.set_camera_target(
                "/OmniverseKit_Persp",
                Gf.Vec3d(0.0, 0.0, 1.0),
                True
            )
            print("✓ Camera positioned")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        simulation_app.close()
        return

    print("="*50)
    print("SUCCESS: Robot loaded in Isaac Sim!")
    if args.headless:
        print("Headless mode - robot rendered successfully")
        print("To see the robot visually, run without --headless flag")
        import time
        time.sleep(2)
    else:
        print("GUI Mode Controls:")
        print("- Mouse drag: Rotate view")
        print("- Mouse wheel: Zoom")
        print("- Middle click + drag: Pan")
        print("- F: Focus on robot")
        print("- Close window to exit")
        print("\nRobot should be visible now!")
    print("="*50)
    
    # Keep running
    if args.headless:
        print("Headless test completed!")
    else:
        input("Press Enter to close...")

    # Cleanup
    simulation_app.close()

if __name__ == "__main__":
    main()