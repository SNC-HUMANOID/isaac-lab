#!/usr/bin/env python3
"""Monitor SNC R5 walking training progress."""

import os
import time
import glob
from pathlib import Path

def monitor_training():
    """Monitor the latest walking training session."""
    
    log_dir = Path("/home/sncbot/IsaacLab/logs/rsl_rl/humanoid_snc_r5_walking")
    
    if not log_dir.exists():
        print("❌ No training logs found")
        return
    
    # Find the latest training session
    sessions = sorted(log_dir.glob("2025-*"))
    if not sessions:
        print("❌ No training sessions found")
        return
    
    latest_session = sessions[-1]
    print(f"🚶 Monitoring Walking Training: {latest_session.name}")
    print("=" * 60)
    
    # Check if training is running
    model_files = sorted(latest_session.glob("model_*.pt"))
    
    if model_files:
        print(f"✅ Found {len(model_files)} saved models")
        print(f"📈 Latest checkpoint: {model_files[-1].name}")
        
        # Estimate progress
        latest_model = model_files[-1].name
        iteration = int(latest_model.replace('model_', '').replace('.pt', ''))
        progress = (iteration / 3000) * 100
        print(f"🎯 Training Progress: {progress:.1f}% ({iteration}/3000 iterations)")
    
    # Check tensorboard logs
    tb_files = list(latest_session.glob("events.out.tfevents.*"))
    if tb_files:
        tb_file = tb_files[0]
        size_mb = tb_file.stat().st_size / (1024*1024)
        print(f"📊 Tensorboard log size: {size_mb:.1f} MB")
        print(f"📍 To view training curves:")
        print(f"    tensorboard --logdir {latest_session}")
    
    # Check configuration
    params_dir = latest_session / "params"
    if params_dir.exists():
        print(f"⚙️  Configuration files saved")
    
    print(f"\n🔄 Training Status:")
    print(f"   - Environment: Isaac-Humanoid-SNC-R5-Walking-Direct-v0")
    print(f"   - Target: Natural walking with coordinated arm movement")
    print(f"   - Key improvements:")
    print(f"     • Forward velocity optimization")
    print(f"     • Natural gait cycle rewards")
    print(f"     • Coordinated arm swing (opposite to legs)")
    print(f"     • Improved torso stability")
    print(f"     • Energy-efficient movement patterns")
    
    print(f"\n📝 Next Steps:")
    print(f"   1. Monitor reward curves in tensorboard")
    print(f"   2. Test intermediate checkpoints with play.py")
    print(f"   3. Compare walking gait with standing-only model")
    print(f"   4. Fine-tune rewards if needed")

if __name__ == "__main__":
    monitor_training()