#!/usr/bin/env python3
"""
Simple G1 Visualization - Easy to use wrapper
ใช้ play script ที่มีอยู่แล้วแทนการเขียนใหม่
"""

import subprocess
import sys
import os

def run_g1_model():
    """Run G1 model with the working command"""
    
    print("🤖 Starting G1 Model Visualization")
    print("=" * 40)
    
    # The command that works (with GUI)
    cmd = [
        "./isaaclab.sh", "-p", 
        "scripts/reinforcement_learning/rsl_rl/play.py",
        "--task", "Isaac-Velocity-Flat-G1-v0",
        "--load_run", "2025-07-29_15-27-26",
        "--num_envs", "1"
        # Removed --headless to show GUI
    ]
    
    print("🚀 Running command:")
    print(" ".join(cmd))
    print()
    print("📋 What you'll see:")
    print("  • Isaac Sim GUI opens (30+ seconds)")
    print("  • Loading G1 environment")
    print("  • Loading trained model")
    print("  • Robot appears and starts moving/walking")
    print("  • You can control camera view with mouse")
    print()
    print("🖱️  Camera Controls:")
    print("  • Left mouse: Rotate view")
    print("  • Right mouse: Pan view") 
    print("  • Scroll wheel: Zoom in/out")
    print()
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 40)
    print()
    
    try:
        # Run the command
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_g1_model()