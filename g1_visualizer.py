#!/usr/bin/env python3
"""
G1 Model Visualizer - Complete GUI and headless support
รันโมเดล G1 พร้อมตัวเลือก GUI หรือ headless
"""

import subprocess
import sys
import os
import argparse

def run_g1_visualization(gui=True, num_envs=1, video=False):
    """Run G1 model visualization"""
    
    print("🤖 G1 Model Visualization")
    print("=" * 50)
    print(f"Mode: {'GUI' if gui else 'Headless'}")
    print(f"Environments: {num_envs}")
    print(f"Video recording: {'Yes' if video else 'No'}")
    print()
    
    # Build command
    cmd = [
        "./isaaclab.sh", "-p", 
        "scripts/reinforcement_learning/rsl_rl/play.py",
        "--task", "Isaac-Velocity-Flat-G1-v0",
        "--load_run", "2025-07-29_15-27-26",
        "--num_envs", str(num_envs)
    ]
    
    # Add headless if no GUI
    if not gui:
        cmd.append("--headless")
    
    # Add video recording
    if video:
        cmd.extend(["--video", "--video_length", "200"])
    
    print("🚀 Running command:")
    print(" ".join(cmd))
    print()
    
    if gui:
        print("📋 GUI Mode - What you'll see:")
        print("  • Isaac Sim window opens (30+ seconds)")
        print("  • G1 robot appears on terrain")
        print("  • Robot starts walking/moving with AI")
        print("  • Real-time 3D visualization")
        print()
        print("🖱️  Camera Controls:")
        print("  • Left mouse drag: Rotate view")
        print("  • Right mouse drag: Pan view")
        print("  • Mouse wheel: Zoom in/out")
        print("  • WASD keys: Move camera")
        print()
        print("⚡ Performance:")
        print("  • GUI mode uses more GPU memory")
        print("  • May run slower than headless")
        print("  • Great for visualization and demos")
        
    else:
        print("📋 Headless Mode - What happens:")
        print("  • No GUI window (background only)")
        print("  • Faster performance")
        print("  • Less GPU memory usage")
        print("  • Good for training/testing")
        
    print()
    print("⏹️  Press Ctrl+C to stop anytime")
    print("=" * 50)
    print()
    
    try:
        # Run the command
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✅ Visualization completed successfully!")
        else:
            print(f"\n⚠️ Process exited with code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    """Main function with command line options"""
    
    parser = argparse.ArgumentParser(
        description="G1 Model Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python g1_visualizer.py              # GUI mode, 1 robot
  python g1_visualizer.py --headless   # No GUI, 1 robot  
  python g1_visualizer.py --num_envs 4 # GUI mode, 4 robots
  python g1_visualizer.py --video      # GUI mode with video recording
        """
    )
    
    parser.add_argument("--headless", action="store_true",
                       help="Run without GUI (faster, less memory)")
    parser.add_argument("--num_envs", type=int, default=1,
                       help="Number of robots (1-8 recommended)")
    parser.add_argument("--video", action="store_true",
                       help="Record video (only works with GUI)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.video and args.headless:
        print("❌ Error: Video recording requires GUI mode")
        print("   Remove --headless or --video")
        sys.exit(1)
    
    if args.num_envs > 8:
        print("⚠️  Warning: More than 8 robots may cause performance issues")
    
    # Run visualization
    run_g1_visualization(
        gui=not args.headless,
        num_envs=args.num_envs, 
        video=args.video
    )

if __name__ == "__main__":
    main()