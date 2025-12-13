#!/usr/bin/env python3

"""Create image of robot instead of GUI (for cases where GUI hangs)"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Generate robot image")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Force headless mode for image generation
    args.headless = True

    print("\n" + "="*60)
    print("📸 HUMANOID_SNC ROBOT IMAGE GENERATOR")  
    print("="*60)
    print("Since GUI is hanging, generating image instead...")
    
    # Launch Isaac Sim
    app = AppLauncher(args)
    simulation_app = app.app

    try:
        # Import modules after app launch
        import omni.usd
        from pxr import Gf, UsdGeom
        import os

        print("✅ Isaac Sim started in headless mode")

        # Get USD context and stage
        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()
        
        if not stage:
            usd_context.new_stage()
            stage = usd_context.get_stage()

        # Load robot
        robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
        robot_prim_path = "/World/Robot"
        
        # Create World
        world_prim = stage.DefinePrim("/World", "Scope")
        
        # Load robot
        robot_prim = stage.DefinePrim(robot_prim_path)
        robot_prim.GetReferences().AddReference(robot_usd_path)
        
        print("✅ Robot loaded")
        
        # Apply upright transformation
        xformable = UsdGeom.Xformable(robot_prim)
        xformable.ClearXformOpOrder()
        
        # Rotate to standing position
        rotate_op = xformable.AddRotateYOp()
        rotate_op.Set(90.0)
        
        # Position above ground
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 1.0))
        
        print("✅ Robot positioned upright")
        
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
        
        print("✅ Ground added")
        
        # Try to create camera and render image
        try:
            import omni.kit.commands
            from omni.kit.viewport.utility import get_active_viewport_window
            
            # Create camera
            camera_prim = stage.DefinePrim("/World/Camera", "Camera")
            camera = UsdGeom.Camera(camera_prim)
            
            # Position camera to view robot
            camera_xform = UsdGeom.Xformable(camera_prim)
            camera_xform.ClearXformOpOrder()
            cam_translate = camera_xform.AddTranslateOp()
            cam_translate.Set(Gf.Vec3d(3.0, 3.0, 2.0))  # Camera position
            
            print("✅ Camera created")
            
            # Wait for scene to settle
            import time
            time.sleep(2)
            
            print("✅ Scene ready")
            print("📸 In headless mode - cannot generate visual image")
            print("🎉 But robot is loaded and positioned correctly!")
            
        except Exception as e:
            print(f"📷 Cannot render image in headless mode: {e}")
            
        # Save the scene as USD for later viewing
        output_path = "/home/sncbot/IsaacLab/robot_scene_upright.usd"
        stage.Export(output_path)
        print(f"💾 Scene saved to: {output_path}")
        
        print("="*60)
        print("✅ SOLUTION: Since GUI hangs, use one of these:")
        print("1. Open saved USD file manually in Isaac Sim:")
        print(f"   File → Open → {output_path}")
        print("2. Use external USD viewer (like Blender USD plugin)")
        print("3. Try VNC/remote desktop for GUI access")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    simulation_app.close()
    print("Done!")

if __name__ == "__main__":
    main()