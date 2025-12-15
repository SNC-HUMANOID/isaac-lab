#!/usr/bin/env python3

"""Train Humanoid SNC with RSL-RL using correct IsaacLab integration."""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train Humanoid SNC with RSL-RL")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import our environment
from simple_train_humanoid_snc import HumanoidSNCEnv, HumanoidSNCEnvCfg


def main():
    """Main training function."""
    
    # Create environment configuration
    env_cfg = HumanoidSNCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Override decimation for training
    env_cfg.decimation = 2  # Control frequency
    env_cfg.sim.dt = 1.0 / 120.0  # Physics frequency
    
    print_dict(env_cfg, nesting=4)
    
    # Create environment
    env = HumanoidSNCEnv(cfg=env_cfg)
    
    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    print(f"[INFO] Environment created with {env.num_envs} environments")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")
    
    # Create algorithm configuration
    agent_cfg = RslRlOnPolicyRunnerCfg()
    
    # Customize training parameters
    agent_cfg.algorithm.learning_rate = 1e-3
    agent_cfg.algorithm.num_learning_epochs = 5
    agent_cfg.algorithm.num_mini_batches = 4
    agent_cfg.algorithm.clip_param = 0.2
    agent_cfg.algorithm.gamma = 0.99
    agent_cfg.algorithm.lam = 0.95
    agent_cfg.algorithm.desired_kl = 0.01
    
    # Policy network configuration
    agent_cfg.policy.actor_hidden_dims = [512, 256, 128]
    agent_cfg.policy.critic_hidden_dims = [512, 256, 128]
    agent_cfg.policy.activation = "elu"
    
    # Setup directories
    log_root_path = os.path.join("logs", "rsl_rl", "humanoid_snc")
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)
    
    # Start training
    print(f"[INFO] Starting training with {args_cli.num_envs} environments")
    print(f"[INFO] Training for {args_cli.max_iterations} iterations")
    print(f"[INFO] Logs will be saved to: {log_dir}")
    
    # Save configurations
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # Run training
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    
    print("[INFO] Training completed!")
    print(f"[INFO] Logs saved to: {log_dir}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run training
    main()
    # Close simulation
    simulation_app.close()