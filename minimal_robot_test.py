#!/usr/bin/env python3

"""Minimal test to check if robot can load"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Minimal robot test")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    print("🔄 Starting minimal robot test...")
    
    # Launch Isaac Sim
    app = AppLauncher(args)
    simulation_app = app.app

    print("✅ Isaac Sim started")

    try:
        # Import USD modules
        import omni.usd
        from pxr import Gf, UsdGeom
        
        print("✅ USD modules imported")

        # Get stage
        usd_context = omni.usd.get_context()
        if not usd_context.get_stage():
            usd_context.new_stage()
        stage = usd_context.get_stage()
        
        print("✅ USD stage ready")

        # Test loading robot
        robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
        robot_prim = stage.DefinePrim("/World/TestRobot")
        robot_prim.GetReferences().AddReference(robot_usd_path)
        
        print("✅ Robot USD reference added")
        
        # Just wait a bit and exit
        import time
        print("⏳ Waiting 3 seconds...")
        time.sleep(3)
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Quick exit
    simulation_app.close()
    print("👋 Done!")

if __name__ == "__main__":
    main()