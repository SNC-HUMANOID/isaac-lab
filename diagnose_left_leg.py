#!/usr/bin/env python3
"""Diagnose why left leg doesn't step."""

import torch
import gymnasium as gym
import numpy as np

# Create minimal environment
env = gym.make("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", num_envs=1)
obs, _ = env.reset()

robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

print("="*80)
print("LEFT LEG DIAGNOSIS")
print("="*80)

# Find joint indices
joints = {}
for i, name in enumerate(robot.data.joint_names):
    if "left" in name or "right" in name:
        if any(x in name for x in ["hip", "knee", "ankle"]):
            joints[name] = i

print(f"\nFound {len(joints)} leg joints:")
for name, idx in sorted(joints.items()):
    print(f"  {idx:2d}: {name}")

# Find contact bodies
print(f"\nContact sensor bodies:")
for i, name in enumerate(contact_sensor.data.body_names):
    if "ankle" in name.lower():
        print(f"  {i}: {name}")

print("\n" + "="*80)
print("TEST 1: Default pose (zero actions)")
print("="*80)

for step in range(100):
    actions = torch.zeros((1, env.action_space.shape[0]))
    obs, reward, terminated, truncated, info = env.step(actions)

if step == 99:
    print("\nJoint positions after 100 steps with zero action:")
    for name in sorted(joints.keys()):
        if "left" in name:
            idx = joints[name]
            pos = robot.data.joint_pos[0, idx].item()
            print(f"  LEFT  {name:30s}: {pos:6.3f}")
    for name in sorted(joints.keys()):
        if "right" in name:
            idx = joints[name]
            pos = robot.data.joint_pos[0, idx].item()
            print(f"  RIGHT {name:30s}: {pos:6.3f}")

print("\n" + "="*80)
print("TEST 2: Force ONLY left knee to bend")
print("="*80)

left_knee_idx = [idx for name, idx in joints.items() if "left_knee" in name][0]
print(f"Left knee joint index: {left_knee_idx}")

env.reset()
for step in range(200):
    actions = torch.zeros((1, env.action_space.shape[0]))
    # Strong positive action on left knee only
    actions[0, left_knee_idx] = 3.0  # Maximum action
    obs, reward, terminated, truncated, info = env.step(actions)

    if step % 50 == 0:
        knee_pos = robot.data.joint_pos[0, left_knee_idx].item()
        knee_vel = robot.data.joint_vel[0, left_knee_idx].item()
        knee_torque = robot.data.applied_torque[0, left_knee_idx].item()
        print(f"Step {step:3d}: pos={knee_pos:6.3f}  vel={knee_vel:6.3f}  torque={knee_torque:7.2f}")

print("\n" + "="*80)
print("TEST 3: Check contact sensor asymmetry")
print("="*80)

env.reset()
for step in range(100):
    actions = torch.randn((1, env.action_space.shape[0])) * 0.5
    obs, reward, terminated, truncated, info = env.step(actions)

print("\nContact sensor readings:")
for i, name in enumerate(contact_sensor.data.body_names):
    if "ankle" in name.lower():
        force = contact_sensor.data.net_forces_w[0, i].norm().item()
        air_time = contact_sensor.data.current_air_time[0, i].item()
        contact_time = contact_sensor.data.current_contact_time[0, i].item()
        print(f"  {name:30s}: force={force:6.1f}N  air={air_time:.2f}s  contact={contact_time:.2f}s")

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

# Check if joints moved in test 2
if abs(robot.data.joint_vel[0, left_knee_idx].item()) < 0.01:
    print("\n❌ PROBLEM: Left knee did NOT move when commanded!")
    print("   Possible causes:")
    print("   1. Actuator not connected properly")
    print("   2. Joint is locked in URDF")
    print("   3. Stiffness too low or damping too high")
    print("   4. Action scale too small")
else:
    print("\n✅ Left knee CAN move")

    # Check contact sensor
    left_ankle_bodies = [i for i, name in enumerate(contact_sensor.data.body_names) if "left" in name.lower() and "ankle" in name.lower()]
    if len(left_ankle_bodies) == 0:
        print("❌ PROBLEM: No left ankle in contact sensor!")
        print("   Contact sensor may not detect left foot")
    else:
        left_contact = contact_sensor.data.current_contact_time[0, left_ankle_bodies[0]].item()
        if left_contact > 5.0:
            print("❌ PROBLEM: Left ankle stuck in contact (>5s)")
            print("   Foot never leaves ground - prevents stepping")
        else:
            print("✅ Contact sensor working")
            print("\n⚠️  Problem may be in REWARD FUNCTION or LEARNING")
            print("   The hardware works but policy doesn't learn to lift left leg")

env.close()
print("\n" + "="*80)
