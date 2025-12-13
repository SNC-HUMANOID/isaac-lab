#!/usr/bin/env python3
"""Test to verify action scale dictionary is working."""

import torch

# Simulate action transformation
print("="*60)
print("ACTION SCALE TEST")
print("="*60)

# Joint names in order
joint_names = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",  # Index 2
    "left_knee_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",  # Index 6
    "right_knee_joint",
]

# Scale dictionary
scale_dict = {
    "left_hip_yaw_joint": 0.5,
    "left_hip_roll_joint": 0.5,
    "left_hip_pitch_joint": -0.5,  # NEGATIVE!
    "left_knee_joint": 0.5,
    "right_hip_yaw_joint": 0.5,
    "right_hip_roll_joint": 0.5,
    "right_hip_pitch_joint": 0.5,  # POSITIVE
    "right_knee_joint": 0.5,
}

# Create scale tensor
scale_tensor = torch.tensor([scale_dict[name] for name in joint_names])

print(f"\nJoint scales:")
for i, name in enumerate(joint_names):
    marker = " ← INVERTED!" if scale_tensor[i] < 0 else ""
    print(f"  [{i}] {name:30s} scale={scale_tensor[i]:+.1f}{marker}")

# Test with sample actions
print(f"\n{'='*60}")
print("TEST: Policy wants both hips to move forward (+1.0)")
print("="*60)

policy_action = torch.ones(8) * 1.0  # All +1.0

scaled_action = policy_action * scale_tensor

print(f"\nPolicy action (raw):")
print(f"  Left hip pitch:  {policy_action[2]:+.2f}")
print(f"  Right hip pitch: {policy_action[6]:+.2f}")

print(f"\nScaled action (to robot):")
print(f"  Left hip pitch:  {scaled_action[2]:+.2f}")
print(f"  Right hip pitch: {scaled_action[6]:+.2f}")

print(f"\nURDF axis effect (Y component):")
print(f"  Left:  {scaled_action[2]:+.2f} × (Y=-0.9999) = {scaled_action[2] * -1:+.2f}")
print(f"  Right: {scaled_action[6]:+.2f} × (Y=+0.9999) = {scaled_action[6] * 1:+.2f}")

print(f"\n{'='*60}")
if abs(scaled_action[2] * -1 - scaled_action[6] * 1) < 0.01:
    print("✅ CORRECT: Both legs will move in SAME direction!")
    print("   After URDF axis, both get same effective movement")
else:
    print("❌ WRONG: Legs will move in DIFFERENT directions")
    print(f"   Left effective:  {scaled_action[2] * -1:+.2f}")
    print(f"   Right effective: {scaled_action[6] * 1:+.2f}")
print("="*60)
