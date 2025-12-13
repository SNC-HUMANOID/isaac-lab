#!/usr/bin/env python3
"""Quick test script to verify humanoid model can walk."""

import argparse
import torch

from isaaclab.app import AppLauncher

# Create parser
parser = argparse.ArgumentParser(description="Test humanoid walking model")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from rsl_rl.runners import OnPolicyRunner

# Import environment configurations
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

def main():
    """Test the trained model."""

    # Get environment configuration
    env_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", "rsl_rl_cfg_entry_point")

    # Set environment parameters
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Create environment
    print(f"\n[INFO] Creating environment: Isaac-Velocity-Flat-Humanoid-SNC-R5-v0")
    env = gym.make("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", cfg=env_cfg)

    # Load the trained model
    print(f"\n[INFO] Loading model from: {args_cli.model_path}")

    # Load checkpoint
    checkpoint = torch.load(args_cli.model_path, map_location=args_cli.device)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)

    # Load the model weights
    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    runner.alg.actor_critic.eval()

    print("\n[INFO] Model loaded successfully!")
    print(f"[INFO] Actor input size: {runner.alg.actor_critic.actor[0].in_features}")
    print(f"[INFO] Actor output size: {runner.alg.actor_critic.actor[-1].out_features}")

    # Test the model
    print("\n[INFO] Running simulation...")
    obs, _ = env.reset()

    total_reward = 0
    episode_length = 0
    max_steps = 500

    for step in range(max_steps):
        with torch.no_grad():
            actions = runner.alg.actor_critic.act_inference(obs["policy"])

        obs, rewards, dones, truncated, info = env.step(actions)
        total_reward += rewards.mean().item()
        episode_length += 1

        if step % 50 == 0:
            print(f"Step {step}/{max_steps} | Avg Reward: {total_reward/episode_length:.3f}")

        if dones.any():
            print(f"\n[INFO] Episode ended at step {step}")
            print(f"[INFO] Total reward: {total_reward:.3f}")
            print(f"[INFO] Average reward: {total_reward/episode_length:.3f}")
            break

    print(f"\n[INFO] Test completed!")
    print(f"[INFO] Final average reward: {total_reward/episode_length:.3f}")

    # Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()