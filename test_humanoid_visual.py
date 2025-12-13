#!/usr/bin/env python
"""Visual test script for trained Humanoid AMP running model"""

import os
import torch
from isaaclab.app import AppLauncher

# Parse arguments (no headless for visual)
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Import after app launch
import isaaclab_tasks
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env_cfg import HumanoidAmpRunEnvCfg

def main():
    print("Starting Humanoid AMP visual test...")
    
    # Create environment with fewer envs for visual
    cfg = HumanoidAmpRunEnvCfg()
    cfg.scene.num_envs = 1  # Single environment for better visualization
    env = HumanoidAmpEnv(cfg=cfg)
    
    print("Environment created successfully!")
    
    # Reset environment
    obs, _ = env.reset()
    print("Environment reset. Running simulation...")
    
    # Run some steps with random actions
    print("Running 500 steps...")
    total_reward = 0
    for step in range(500):
        # Small random actions to see some movement
        actions = 0.1 * torch.randn(env.num_envs, env.cfg.action_space, device=env.device)
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += rewards.mean().item()
        
        if step % 50 == 0:
            print(f"Step {step}: Reward = {rewards.mean().item():.4f}")
    
    print(f"Average reward: {total_reward/500:.4f}")
    print("Test completed! The simulation should be visible.")
    
    # Keep simulation running for observation
    input("Press Enter to close simulation...")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()