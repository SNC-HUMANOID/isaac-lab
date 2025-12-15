#!/usr/bin/env python3

"""Debug GUI issues with Isaac Sim"""

import argparse
from isaaclab.app import AppLauncher
import os

def main():
    parser = argparse.ArgumentParser(description="Debug Isaac Sim GUI issues")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Force specific display settings
    os.environ['DISPLAY'] = ':0'
    os.environ['QT_X11_NO_MITSHM'] = '1'
    
    print("\n" + "="*50)
    print("🔍 ISAAC SIM GUI DEBUG")
    print("="*50)
    print("Environment check:")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")
    print(f"Headless: {args.headless}")
    print(f"X11 available: {os.path.exists('/tmp/.X11-unix')}")
    
    # Launch Isaac Sim with minimal setup
    app = AppLauncher(args)
    simulation_app = app.app

    print("✅ Isaac Sim App started")
    print("⏳ Waiting for GUI to initialize...")

    try:
        # Import modules after launch
        import omni.usd
        print("✅ USD modules loaded")

        # Get USD context
        usd_context = omni.usd.get_context()
        if not usd_context.get_stage():
            usd_context.new_stage()
        
        print("✅ USD stage ready")
        
        # Simple test - create a cube
        stage = usd_context.get_stage()
        from pxr import UsdGeom, Gf
        
        cube_prim = stage.DefinePrim("/World/TestCube", "Cube")
        cube = UsdGeom.Cube(cube_prim)
        cube.CreateSizeAttr(1.0)
        
        print("✅ Test cube created")
        
        if args.headless:
            print("ℹ️  Running in headless mode - no GUI expected")
            import time
            time.sleep(3)
            print("✅ Headless test completed")
        else:
            print("🖥️  GUI mode - checking if window appears...")
            print("If you don't see Isaac Sim window:")
            print("1. Check if it's minimized")
            print("2. Try Alt+Tab to switch windows")
            print("3. Check system tray")
            print("")
            print("Press Enter to continue or Ctrl+C to quit...")
            
            # Keep running with timeout
            import signal
            import time
            
            def timeout_handler(signum, frame):
                print("\n⏰ Timeout reached - closing Isaac Sim")
                raise KeyboardInterrupt
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 second timeout
            
            try:
                input()
                signal.alarm(0)  # Cancel timeout
            except KeyboardInterrupt:
                print("\nClosing...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("Shutting down Isaac Sim...")
    simulation_app.close()
    print("✅ Done")

if __name__ == "__main__":
    main()