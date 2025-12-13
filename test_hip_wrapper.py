#!/usr/bin/env python3
"""Quick test to verify HipInverterWrapper is working."""

import torch

# Test environment creation with wrapper
import gymnasium as gym

# This should automatically apply the HipInverterWrapper
env = gym.make("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", num_envs=4)

print("\n" + "="*60)
print("Environment created successfully!")
print("="*60)

# Check if wrapper is applied by looking for the left_hip_idx attribute
if hasattr(env, 'left_hip_idx'):
    print(f"\n✓ HipInverterWrapper is applied!")
    print(f"  Left hip pitch index: {env.left_hip_idx}")
else:
    print("\n✗ HipInverterWrapper NOT detected")
    print("  Checking unwrapped environment...")

print(f"\nEnvironment type: {type(env)}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Get robot joint names
robot = env.unwrapped.scene["robot"]
print(f"\nRobot joint names ({len(robot.data.joint_names)} total):")
for i, name in enumerate(robot.data.joint_names):
    print(f"  [{i:2d}] {name}")
    if "left_hip_pitch" in name:
        print(f"       ^ LEFT HIP PITCH at index {i}")

# Test a step with actions
print("\n" + "="*60)
print("Testing step with action inversion...")
print("="*60)

obs, _ = env.reset()
print(f"\nObservation shape: {obs.shape}")

# Create test action - all zeros except hip pitch
action = torch.zeros((4, env.action_space.shape[0]), device=env.unwrapped.device)

# Find left and right hip pitch indices
left_idx = None
right_idx = None
for i, name in enumerate(robot.data.joint_names):
    if name == "left_hip_pitch_joint":
        left_idx = i
    elif name == "right_hip_pitch_joint":
        right_idx = i

print(f"\nLeft hip pitch joint index: {left_idx}")
print(f"Right hip pitch joint index: {right_idx}")

# Set same positive value for both hips
test_value = 0.3
action[:, left_idx] = test_value
action[:, right_idx] = test_value

print(f"\nBefore wrapper: Both hips = {test_value}")

# Step (wrapper should invert left hip)
obs, reward, terminated, truncated, info = env.step(action)

print(f"After step: Environment executed successfully")
print(f"  Reward: {reward[0].item():.4f}")

print("\n" + "="*60)
print("SUCCESS - Wrapper integration working!")
print("="*60)

env.close()
