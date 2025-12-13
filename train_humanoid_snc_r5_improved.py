#!/usr/bin/env python3
"""Training script for Humanoid SNC R5 with improved walking configuration."""

import argparse
from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Train Humanoid SNC R5 for walking")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=2000, help="Max training iterations")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching the app
import gymnasium as gym
import os
import torch
from datetime import datetime

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

def main():
    """Train the Humanoid SNC R5 robot."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        "Isaac-Velocity-Flat-Humanoid-SNC-R5-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Set seed
    env_cfg.seed = args_cli.seed

    # Create environment
    print(f"\n{'='*80}")
    print(f"Training Humanoid SNC R5 - Improved Walking Configuration")
    print(f"{'='*80}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Device: {args_cli.device}")
    print(f"Seed: {args_cli.seed}")
    print(f"Max iterations: {args_cli.max_iterations}")
    print(f"{'='*80}\n")

    env = gym.make("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", cfg=env_cfg)

    # Import agent configuration
    from isaaclab_tasks.manager_based.locomotion.velocity.config.humanoid_snc_r5.agents import rsl_rl_ppo_cfg

    agent_cfg = rsl_rl_ppo_cfg.HumanoidSNCR5PPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = 50  # Save every 50 iterations

    # Adjust learning parameters for better walking
    agent_cfg.algorithm.learning_rate = 3e-4
    agent_cfg.algorithm.num_learning_epochs = 5
    agent_cfg.algorithm.num_mini_batches = 8

    # Create log directory
    log_root_path = os.path.join("logs", "rsl_rl", "humanoid_snc_r5_walking_improved")
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    print(f"Logging to: {log_dir}\n")

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args_cli.device)

    # Train
    print("\nStarting training...\n")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Close environment
    env.close()

    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Model saved to: {log_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
    simulation_app.close()