#!/usr/bin/env python3
"""
Quick test script for the improved balance environment
"""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Humanoid Balance Environment")
parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_balance_env import HumanoidSNCBalanceEnv, HumanoidSNCBalanceEnvCfg

def main():
    print("🧪 Testing Humanoid Balance Environment")

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"Device: {device}")

    # Create config
    env_cfg = HumanoidSNCBalanceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    print("Creating environment...")
    try:
        env = HumanoidSNCBalanceEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)
        print(f"✅ Environment created successfully!")
        print(f"   - Environments: {env.num_envs}")
        print(f"   - Actions: {env.num_actions}")
        print(f"   - Observation space: {env.observation_space.shape}")
        print(f"   - Action space: {env.action_space.shape}")

        # Test reset
        print("\nTesting reset...")
        obs, info = env.reset()
        if isinstance(obs, dict):
            obs_tensor = obs["policy"]
            print(f"✅ Reset successful! Obs dict keys: {list(obs.keys())}, shape: {obs_tensor.shape}")
        else:
            print(f"✅ Reset successful! Obs shape: {obs.shape}")

        # Test step
        print("\nTesting random actions...")
        for i in range(10):
            actions = torch.randn(env.num_envs, env.num_actions, device=device) * 0.1
            obs, rewards, dones, info = env.step(actions)
            print(f"Step {i+1}: Reward mean={rewards.mean():.3f}, Done={dones.sum().item()}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()