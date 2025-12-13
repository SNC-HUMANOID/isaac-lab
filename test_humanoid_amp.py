#!/usr/bin/env python
"""Test script for trained Humanoid AMP running model"""

import os
import torch
import pickle
from isaaclab.app import AppLauncher

# Parse arguments
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Import after app launch
import isaaclab_tasks
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env_cfg import HumanoidAmpRunEnvCfg

def main():
    # Create environment
    cfg = HumanoidAmpRunEnvCfg()
    cfg.scene.num_envs = 4
    env = HumanoidAmpEnv(cfg=cfg)
    
    # Load trained model
    checkpoint_path = "/home/sncbot/IsaacLab/logs/skrl/humanoid_amp_run/2025-08-09_14-14-26_ppo_torch/checkpoints/best_agent.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=env.device)
        print("Model loaded successfully!")
        
        # Get model state
        if 'policy' in checkpoint:
            print("Found policy in checkpoint")
        if 'state_dict' in checkpoint:
            print("Found state_dict in checkpoint")
        
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
    else:
        print(f"Checkpoint not found at: {checkpoint_path}")
        return
    
    # Reset environment
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_policy = obs["policy"]
        print(f"Environment reset. Policy observations shape: {obs_policy.shape}")
    else:
        print(f"Environment reset. Observations shape: {obs.shape}")
    
    # Run some steps with random actions
    print("Running 1000 steps with random actions...")
    total_reward = 0
    for step in range(1000):
        actions = torch.randn(env.num_envs, env.cfg.action_space, device=env.device)
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += rewards.mean().item()
        
        if step % 100 == 0:
            print(f"Step {step}: Mean reward = {rewards.mean().item():.4f}, Terminated = {terminated.sum().item()}/{env.num_envs}")
    
    print(f"Average reward over 1000 steps: {total_reward/1000:.4f}")
    
    print("Test completed!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()