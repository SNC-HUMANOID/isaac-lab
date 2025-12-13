#!/usr/bin/env python3

"""Quick and simple robot viewer"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim 
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    print("Isaac Sim started. Loading robot...")

    # Import modules after launch
    import omni.usd
    import omni.kit.commands
    from omni.isaac.core.utils.stage import open_stage

    try:
        # Simply open the USD file directly  
        robot_usd_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
        
        print(f"Opening USD file: {robot_usd_path}")
        open_stage(usd_path=robot_usd_path)
        
        print("✅ Robot USD file opened!")
        print("🎉 You should see the Humanoid_SNC robot now!")
        
        if not args.headless:
            print("\nControls:")
            print("- Mouse: Rotate view")
            print("- Wheel: Zoom")
            print("- F: Focus on robot")
            print("- Close window or Ctrl+C to exit")
            
            # Keep running until user closes
            while simulation_app.is_running():
                simulation_app.update()
        else:
            import time
            time.sleep(5)
            print("Headless mode completed")
            
    except Exception as e:
        print(f"Error: {e}")

    simulation_app.close()

if __name__ == "__main__":
    main()