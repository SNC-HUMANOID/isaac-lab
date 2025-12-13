#!/usr/bin/env python3

"""Test script for Humanoid SNC with simple random policy."""

import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401

def main():
    """Test the Humanoid SNC environment with random actions."""
    
    # Create environment
    env = gym.make("Isaac-Humanoid-SNC-Direct-v0", num_envs=1)
    
    print(f"Environment: {env}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    observations, _ = env.reset()
    print(f"Initial observations shape: {observations.shape}")
    
    # Run for 1000 steps with random actions
    for i in range(1000):
        # Sample random actions
        actions = env.action_space.sample()
        
        # Apply actions and get next observations
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        if i % 100 == 0:
            print(f"Step {i}: Reward = {rewards.mean():.3f}")
        
        # Reset if needed
        if terminated.any() or truncated.any():
            observations, _ = env.reset()
            print(f"Reset at step {i}")
    
    print("Test completed successfully!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()