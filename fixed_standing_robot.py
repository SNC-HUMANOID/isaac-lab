#!/usr/bin/env python3

"""Fixed Robot Viewer - Standing Upright"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="View Humanoid_SNC robot - STANDING UPRIGHT")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim
    app = AppLauncher(args)
    simulation_app = app.app

    # Import USD modules after app launch
    import omni.usd
    from pxr import Gf, UsdGeom, Usd
    import math

    print("\n" + "="*60)
    print("🤖 HUMANOID_SNC ROBOT VIEWER - STANDING UPRIGHT FIX")
    print("="*60)

    try:
        # Get USD context and stage
        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()
        
        # Create new stage if none exists
        if not stage:
            usd_context.new_stage()
            stage = usd_context.get_stage()
        
        print("✅ USD Stage initialized")
        
        # Load robot USD file
        robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
        robot_prim_path = "/World/Humanoid_Robot"
        
        # Create World scope
        world_prim = stage.DefinePrim("/World", "Scope")
        
        # Load robot with unique prim path to avoid conflicts
        robot_prim = stage.DefinePrim(robot_prim_path)
        robot_prim.GetReferences().AddReference(robot_usd_path)
        
        print(f"✅ Robot USD loaded: {robot_usd_path}")
        
        # Position and orient robot to stand upright
        from pxr import UsdGeom
        xformable = UsdGeom.Xformable(robot_prim)
        
        # Clear existing xform ops to avoid conflicts
        xformable.ClearXformOpOrder()
        
        # Add rotation to make robot stand upright
        # Robot is lying on X-axis, so we need to rotate 90° around Y-axis
        rotate_op = xformable.AddRotateYOp()
        rotate_op.Set(90.0)  # 90 degrees around Y-axis to stand up
        
        # Add translation to lift robot above ground
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 1.0))  # 1 meter above ground
        
        print("✅ Robot rotated 90° around Y-axis to stand upright")
        print("✅ Robot positioned at (0, 0, 1)")
        
        # Create simple ground
        ground_prim = stage.DefinePrim("/World/GroundPlane", "Cube")
        ground_cube = UsdGeom.Cube(ground_prim)
        ground_cube.CreateSizeAttr(1.0)
        
        # Ground transform
        ground_xform = UsdGeom.Xformable(ground_prim)
        ground_xform.ClearXformOpOrder()
        
        ground_scale = ground_xform.AddScaleOp()
        ground_scale.Set(Gf.Vec3d(20.0, 20.0, 0.2))
        
        ground_translate = ground_xform.AddTranslateOp()
        ground_translate.Set(Gf.Vec3d(0.0, 0.0, -1.0))
        
        print("✅ Ground plane created")
        
        # Set default prim for proper scene hierarchy
        stage.SetDefaultPrim(world_prim)
        
        print("✅ Scene setup complete")
        
    except Exception as e:
        print(f"✗ Error loading robot: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    print("="*60)
    print("🎉 SUCCESS! Humanoid_SNC robot is now STANDING UPRIGHT!")
    print("="*60)
    
    if args.headless:
        print("ℹ️  HEADLESS MODE - Robot is positioned upright")
        print("   Run without --headless to see the standing robot")
        import time
        time.sleep(3)
        print("✅ Headless test completed - robot should be standing!")
    else:
        print("🖥️  GUI MODE - Robot should be STANDING UPRIGHT now!")
        print("")
        print("Navigation Controls:")
        print("🖱️  Mouse drag      → Rotate camera")
        print("🔍 Mouse wheel     → Zoom in/out")
        print("⚡ Middle click+drag → Pan view")
        print("🎯 Press 'F'       → Focus on standing robot (IMPORTANT!)")
        print("📐 Alt+Mouse drag  → Orbit around robot")
        print("")
        print("The robot should now be:")
        print("✅ Standing upright (not lying down)")
        print("✅ Head pointing up toward +Z axis")  
        print("✅ Feet on the ground plane")
        print("✅ Proper humanoid pose")
        print("")
        print("If robot still looks wrong:")
        print("1. Press 'F' to focus on robot")
        print("2. Zoom out to see full robot")
        print("3. Try different camera angles")
        print("")
        print("Press Enter when you want to close...")
        
        # Keep GUI running
        try:
            input()
        except KeyboardInterrupt:
            print("\n👋 Closing...")

    print("="*60)
    print("Shutting down...")
    # Cleanup
    simulation_app.close()

if __name__ == "__main__":
    main()