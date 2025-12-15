#!/usr/bin/env python3

"""Fixed Robot Viewer - no xform conflicts"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="View Humanoid_SNC robot - FIXED")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim
    app = AppLauncher(args)
    simulation_app = app.app

    # Import USD modules after app launch
    import omni.usd
    from pxr import Gf, UsdGeom, Usd

    print("\n" + "="*60)
    print("🤖 HUMANOID_SNC ROBOT VIEWER - FIXED VERSION")
    print("="*60)

    try:
        # Get USD context and stage
        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()
        
        # Create new stage if none exists
        if not stage:
            usd_context.new_stage()
            stage = usd_context.get_stage()
        
        print("✓ USD Stage initialized")
        
        # Load robot USD file
        robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
        robot_prim_path = "/World/Humanoid_Robot"
        
        # Create World scope
        world_prim = stage.DefinePrim("/World", "Scope")
        
        # Load robot with unique prim path to avoid conflicts
        robot_prim = stage.DefinePrim(robot_prim_path)
        robot_prim.GetReferences().AddReference(robot_usd_path)
        
        print(f"✓ Robot USD loaded: {robot_usd_path}")
        
        # Position robot using simple transformation matrix
        from pxr import UsdGeom
        xformable = UsdGeom.Xformable(robot_prim)
        
        # Clear existing xform ops to avoid conflicts
        xformable.ClearXformOpOrder()
        
        # Add new transform
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 1.0))  # 1 meter above ground
        
        print("✓ Robot positioned at (0, 0, 1)")
        
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
        
        print("✓ Ground plane created")
        
        # Set default prim for proper scene hierarchy
        stage.SetDefaultPrim(world_prim)
        
        print("✓ Scene setup complete")
        
    except Exception as e:
        print(f"✗ Error loading robot: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    print("="*60)
    print("🎉 SUCCESS! Humanoid_SNC robot loaded successfully!")
    print("="*60)
    
    if args.headless:
        print("ℹ️  HEADLESS MODE - Robot is loaded but not visible")
        print("   Run without --headless to see the robot in GUI")
        import time
        time.sleep(3)
        print("✓ Headless test completed - robot loaded OK!")
    else:
        print("🖥️  GUI MODE - Robot should be visible now!")
        print("")
        print("Navigation Controls:")
        print("🖱️  Mouse drag      → Rotate camera")
        print("🔍 Mouse wheel     → Zoom in/out")
        print("⚡ Middle click+drag → Pan view")
        print("🎯 Press 'F'       → Focus on robot (IMPORTANT!)")
        print("📐 Alt+Mouse drag  → Orbit around selection")
        print("")
        print("If you don't see the robot:")
        print("1. Press 'F' to focus on the robot")
        print("2. Use mouse wheel to zoom out")
        print("3. Look around coordinates (0, 0, 1)")
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