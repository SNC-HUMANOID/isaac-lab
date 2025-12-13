#!/usr/bin/env python3
"""Quick test to check if Humanoid 10102025 environment can be loaded."""

import argparse

# Launch Isaac Sim (Omniverse Kit) before importing modules that rely on omni.*
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# Import isaaclab_tasks to register environments (after AppLauncher so omni.log is available)
import isaaclab_tasks  # registers tasks
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.humanoid_10102025.flat_env_cfg import (
    Humanoid10102025FlatEnvCfg,
)

# Test environment loading
try:
    print("Testing Isaac-Velocity-Flat-Humanoid-10102025-v0...")
    cfg = Humanoid10102025FlatEnvCfg()
    env = ManagerBasedRLEnv(cfg)
    print("✅ Environment constructed successfully!")
    
    # Try to reset
    print("\nTesting environment reset...")
    obs, info = env.reset()
    print(f"✅ Environment reset successful!")
    try:
        print(f"   Observation shape: {getattr(obs, 'shape', 'N/A')}")
    except Exception:
        pass
    
    # Try a step
    print("\nTesting environment step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Environment step successful!")
    
    env.close()
    # Close simulator cleanly
    simulation_app.close()
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    # ensure simulator closed on failure
    try:
        try:
            simulation_app.close()
        except Exception:
            pass
    except Exception:
        pass
