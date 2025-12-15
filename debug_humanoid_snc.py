#!/usr/bin/env python3
"""Debug script to check for NaN/Inf values in humanoid_snc environment"""

import torch
import sys
sys.path.append('/home/sncbot/IsaacLab/source')

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_env import HumanoidSNCEnvCfg, LocomotionEnv

def check_tensor_validity(tensor, name):
    """Check if tensor has NaN or Inf values"""
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}: {tensor}")
        return False
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}: {tensor}")
        return False
    return True

def main():
    # Create environment
    cfg = HumanoidSNCEnvCfg()
    cfg.scene.num_envs = 1
    env = LocomotionEnv(cfg)
    
    print(f"Environment created with {env.num_envs} environments")
    print(f"Observation space: {env.single_observation_space}")
    print(f"Action space: {env.single_action_space}")
    
    # Reset environment
    obs, info = env.reset()
    policy_obs = obs["policy"]
    print(f"Initial observation shape: {policy_obs.shape}")
    print(f"Observation range: [{policy_obs.min():.4f}, {policy_obs.max():.4f}]")
    
    # Check for invalid values
    valid = check_tensor_validity(policy_obs, "initial observations")
    if not valid:
        print("Invalid values found in initial observations!")
        return
    
    # Run a few steps with random actions
    for step in range(10):
        actions = torch.randn(env.num_envs, env.single_action_space.shape[0], device=env.device)
        
        print(f"\nStep {step}:")
        print(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")
        
        obs, reward, terminated, truncated, info = env.step(actions)
        policy_obs = obs["policy"]
        
        print(f"Observation range: [{policy_obs.min():.4f}, {policy_obs.max():.4f}]")
        print(f"Reward: {reward.item():.4f}")
        
        # Check for invalid values
        valid_obs = check_tensor_validity(policy_obs, f"step {step} observations")
        valid_reward = check_tensor_validity(reward, f"step {step} reward")
        
        if not valid_obs or not valid_reward:
            print(f"Invalid values found at step {step}!")
            break
    
    print("Debug completed successfully!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()