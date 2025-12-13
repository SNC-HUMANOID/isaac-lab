#!/usr/bin/env python3

"""Test different robot orientations to find the correct upright pose"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Test robot orientations")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim
    app = AppLauncher(args)
    simulation_app = app.app

    # Import USD modules after app launch
    import omni.usd
    from pxr import Gf, UsdGeom, Usd

    print("\n" + "="*60)
    print("🔄 TESTING ROBOT ORIENTATIONS")
    print("="*60)

    try:
        # Get USD context and stage
        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()
        
        if not stage:
            usd_context.new_stage()
            stage = usd_context.get_stage()
        
        # Create World
        world_prim = stage.DefinePrim("/World", "Scope")
        
        robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
        
        # Test multiple orientations
        orientations = [
            ("Original", (0, 0, 0), (-2, 0, 1)),
            ("Rotate_Y_90", (0, 90, 0), (0, 0, 1)),
            ("Rotate_Y_-90", (0, -90, 0), (2, 0, 1)),
            ("Rotate_X_90", (90, 0, 0), (0, -2, 1)),
            ("Rotate_X_-90", (-90, 0, 0), (0, 2, 1)),
            ("Rotate_Z_90", (0, 0, 90), (0, 0, 1)),
        ]
        
        for i, (name, (rx, ry, rz), (tx, ty, tz)) in enumerate(orientations):
            robot_prim_path = f"/World/Robot_{name}"
            
            # Load robot
            robot_prim = stage.DefinePrim(robot_prim_path)
            robot_prim.GetReferences().AddReference(robot_usd_path)
            
            # Apply transformation
            xformable = UsdGeom.Xformable(robot_prim)
            xformable.ClearXformOpOrder()
            
            # Add rotations
            if rx != 0:
                rotate_x = xformable.AddRotateXOp()
                rotate_x.Set(rx)
            if ry != 0:
                rotate_y = xformable.AddRotateYOp()
                rotate_y.Set(ry)
            if rz != 0:
                rotate_z = xformable.AddRotateZOp()
                rotate_z.Set(rz)
            
            # Add translation
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(tx, ty, tz))
            
            print(f"✅ Created {name}: rotation({rx}, {ry}, {rz}), position({tx}, {ty}, {tz})")
        
        # Add ground
        ground_prim = stage.DefinePrim("/World/Ground", "Cube")
        ground_cube = UsdGeom.Cube(ground_prim)
        ground_cube.CreateSizeAttr(1.0)
        
        ground_xform = UsdGeom.Xformable(ground_prim)
        ground_xform.ClearXformOpOrder()
        ground_scale = ground_xform.AddScaleOp()
        ground_scale.Set(Gf.Vec3d(10.0, 10.0, 0.1))
        ground_translate = ground_xform.AddTranslateOp()
        ground_translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        
        stage.SetDefaultPrim(world_prim)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        simulation_app.close()
        return

    print("="*60)
    print("🎉 MULTIPLE ROBOT ORIENTATIONS LOADED!")
    print("="*60)
    print("You should now see 6 robots in different orientations:")
    print("1. Original - lying on X-axis")
    print("2. Rotate_Y_90 - should be standing upright")
    print("3. Rotate_Y_-90 - standing but facing opposite")
    print("4. Rotate_X_90 - rotated around X")
    print("5. Rotate_X_-90 - rotated around X (opposite)")
    print("6. Rotate_Z_90 - rotated around Z")
    print("")
    print("Find which orientation looks correct for standing humanoid!")
    print("Press F to focus, use mouse to explore all robots")
    print("="*60)
    
    if not args.headless:
        try:
            input("Press Enter to close...")
        except KeyboardInterrupt:
            print("\n👋 Closing...")

    simulation_app.close()

if __name__ == "__main__":
    main()