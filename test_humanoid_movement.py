#!/usr/bin/env python3
"""
🔍 Test Humanoid Movement
Quick test to see if robot moves at all
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Humanoid Movement")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_env import HumanoidSNCEnv, HumanoidSNCEnvCfg

def main():
    print("🔍 Testing Humanoid Movement")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create test environment
    env_cfg = HumanoidSNCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.episode_length_s = 10.0
    env_cfg.action_scale = 0.8  # Allow strong actions

    print("Creating environment...")
    env = HumanoidSNCEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(f"✅ Environment created!")
    print(f"   - Environments: {env.num_envs}")
    print(f"   - Actions: {env.num_actions}")

    print("\n🎯 Testing movement patterns...")

    try:
        # Reset environment
        obs, info = env.reset()
        print(f"Initial reset done. Obs shape: {obs.shape if hasattr(obs, 'shape') else 'dict'}")

        # Test different movement patterns
        patterns = [
            ("Standing still", lambda: torch.zeros(env.num_envs, env.num_actions, device=device)),
            ("Walking pattern", lambda: torch.tensor([
                # Arms
                0.0, 0.2, 0.0, -0.3,  # left arm
                0.0, -0.2, 0.0, -0.3, # right arm
                0.0,                   # torso
                # Legs - walking pattern
                0.3, 0.0, 0.0, -0.6, 0.3, 0.0,  # left leg
                -0.3, 0.0, 0.0, 0.6, -0.3, 0.0  # right leg
            ], device=device).repeat(env.num_envs, 1)),
            ("Forward lean", lambda: torch.tensor([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2,  # lean forward
                0.1, 0.0, 0.0, -0.2, 0.1, 0.0,  # left leg
                0.1, 0.0, 0.0, -0.2, 0.1, 0.0   # right leg
            ], device=device).repeat(env.num_envs, 1)),
            ("Random actions", lambda: torch.randn(env.num_envs, env.num_actions, device=device) * 0.3),
        ]

        for pattern_name, action_fn in patterns:
            print(f"\n--- Testing: {pattern_name} ---")

            # Reset for each test
            obs, info = env.reset()

            for step in range(50):  # 50 steps per test
                actions = action_fn()

                # Ensure correct action shape
                if actions.shape != (env.num_envs, env.num_actions):
                    print(f"Action shape mismatch: {actions.shape} vs expected {(env.num_envs, env.num_actions)}")
                    continue

                obs, rewards, dones, info = env.step(actions)

                if step % 10 == 0:  # Print every 10 steps
                    avg_reward = rewards.mean().item()
                    avg_height = torch.mean(env.torso_position[:, 2]).item() if hasattr(env, 'torso_position') else 0
                    num_fallen = dones.sum().item()

                    print(f"  Step {step:2d}: Reward={avg_reward:6.3f}, Height={avg_height:.2f}m, Fallen={num_fallen}")

                    # Check if robot is moving
                    if hasattr(env, 'velocity'):
                        avg_speed = torch.norm(env.velocity[:, :2], dim=1).mean().item()
                        print(f"           Speed={avg_speed:.3f} m/s")

                # Reset fallen robots
                if dones.any():
                    fallen_ids = torch.nonzero(dones, as_tuple=False).flatten()
                    if len(fallen_ids) > 0:
                        print(f"           🚨 {len(fallen_ids)} robots fell, resetting...")

        print(f"\n🎉 Movement testing completed!")
        print("Check the console output above to see:")
        print("  - Which patterns work best")
        print("  - Robot speed and movement")
        print("  - How often robots fall")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()