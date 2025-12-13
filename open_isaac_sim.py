#!/usr/bin/env python3

"""Open Isaac Sim GUI for manual robot loading"""

from isaaclab.app import AppLauncher
import argparse

def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Force GUI mode (no headless)
    args.headless = False
    
    print("\n" + "="*70)
    print("🤖 ISAAC SIM - MANUAL ROBOT LOADING")
    print("="*70)
    print("🚀 Starting Isaac Sim GUI...")
    print("⏳ This may take 30-60 seconds to load")
    print("="*70)
    
    # Launch Isaac Sim in GUI mode
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    print("✅ Isaac Sim is starting...")
    
    # Wait for Isaac Sim to fully load
    import time
    time.sleep(5)
    
    print("\n" + "="*70)
    print("📖 MANUAL INSTRUCTIONS TO LOAD YOUR ROBOT:")
    print("="*70)
    print("1. Wait for Isaac Sim GUI to fully load")
    print("2. In the menu bar, click: File → Open")
    print("3. Navigate to this file:")
    print("   /home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd")
    print("4. Click 'Open' to load the robot")
    print("5. Press 'F' to focus on the robot")
    print("6. Use mouse to rotate and zoom to see the robot")
    print("="*70)
    print("🎮 Controls:")
    print("   - Mouse drag: Rotate view")
    print("   - Mouse wheel: Zoom in/out")
    print("   - F key: Focus on robot")
    print("   - Close window to exit")
    print("="*70)
    
    try:
        # Keep Isaac Sim running
        while simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.016)  # ~60 FPS
    except KeyboardInterrupt:
        print("\n👋 Closing Isaac Sim...")
    
    simulation_app.close()
    print("Isaac Sim closed.")

if __name__ == "__main__":
    main()