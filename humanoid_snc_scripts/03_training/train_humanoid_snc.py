#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for Humanoid SNC robot using RSL-RL.

Usage:
    python train_humanoid_snc.py --num_envs 1024 --headless
"""

import argparse
import sys
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train Humanoid SNC with RSL-RL")
parser.add_argument("--video", action="store_true", help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of video to record.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Humanoid-SNC-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# RSL-RL specific arguments  
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of training iterations.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

# Import Isaac Lab modules
from isaaclab.utils.dict import print_dict

# Import our custom environment
from humanoid_snc_locomotion import HumanoidSNCEnvCfg, HumanoidSNCEnv

gym.register(
    id="Isaac-Humanoid-SNC-v0",
    entry_point="humanoid_snc_locomotion:HumanoidSNCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "humanoid_snc_locomotion:HumanoidSNCEnvCfg",
    },
)


def main():
    """Main training function."""
    # Parse configuration
    env_cfg: HumanoidSNCEnvCfg = HumanoidSNCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"
    
    # Print environment configuration
    print_dict(env_cfg, nesting=4)
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "rsl_rl", "humanoid_snc_locomotion", "exported", "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos during training: {video_kwargs['video_folder']}")
        print(f"[INFO] Video length: {args_cli.video_length}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Create RSL-RL agent configuration
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
    from rsl_rl.env import VecEnv
    
    # RSL-RL configuration - correct format
    from rsl_rl.algorithms.ppo import PPO
    from rsl_rl.modules.actor_critic import ActorCritic
    
    # Create algorithm config dict
    agent_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 1e-3,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
        },
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        }
    }
    
    # Create algorithm configuration
    from rsl_rl.algorithms.ppo import PPO
    from rsl_rl.modules.actor_critic import ActorCritic
    
    # Setup directories
    log_root_path = os.path.join("logs", "rsl_rl", "humanoid_snc_locomotion")
    log_dir = os.path.join(log_root_path, "summaries")
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    
    # The environment is already compatible with RSL-RL
    # No need to wrap with VecEnv
    
    # Create runner
    runner = OnPolicyRunner(env.unwrapped, agent_cfg, log_dir=log_dir, device=env_cfg.sim.device)
    
    # Set seed if provided
    if args_cli.seed is not None:
        runner.set_seed(args_cli.seed)
    
    # Start training
    print(f"[INFO] Starting training with {args_cli.num_envs} environments for {args_cli.max_iterations} iterations")
    
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    
    print("[INFO] Training completed!")


if __name__ == "__main__":
    # Run training
    main()
    # Close simulation
    simulation_app.close()