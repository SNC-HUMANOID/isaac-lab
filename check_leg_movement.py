#!/usr/bin/env python3
"""Check if both legs can actually move."""

import torch

# Simple check without Isaac Sim
print("="*80)
print("CHECKING LEG MOVEMENT CAPABILITY")
print("="*80)

# Check joint patterns in config
import sys
sys.path.append('/home/sncbot/IsaacLab/source/isaaclab_assets')
from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG

print("\n1. INIT STATE CHECK:")
print("-" * 80)
init_joints = HUMANOID_SNC_R5_CFG.init_state.joint_pos
for key, value in init_joints.items():
    if 'hip' in key or 'knee' in key or 'ankle' in key:
        print(f"  {key:40s}: {value}")

print("\n2. ACTUATOR CHECK:")
print("-" * 80)
for name, actuator in HUMANOID_SNC_R5_CFG.actuators.items():
    print(f"\nActuator group: {name}")
    print(f"  Joint patterns: {actuator.joint_names_expr}")
    if hasattr(actuator, 'stiffness'):
        if isinstance(actuator.stiffness, dict):
            for jname, stiff in actuator.stiffness.items():
                if any(x in jname for x in ['hip', 'knee', 'ankle']):
                    print(f"    {jname:40s}: stiffness={stiff}")

print("\n3. PATTERN MATCHING TEST:")
print("-" * 80)

test_joints = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint"
]

import re
for pattern_key in init_joints.keys():
    if '.*' in pattern_key:
        pattern = pattern_key.replace('.*', '.*?')
        print(f"\nPattern: {pattern_key}")
        for joint in test_joints:
            if re.match(pattern, joint):
                print(f"  ✓ Matches: {joint}")

print("\n4. SYMMETRY CHECK:")
print("-" * 80)

# Check if left and right get same values from patterns
left_matches = {}
right_matches = {}

for pattern_key, value in init_joints.items():
    if '.*' in pattern_key:
        pattern = pattern_key.replace('.*', '.*?')
        for joint in test_joints:
            if re.match(pattern, joint):
                if 'left' in joint:
                    left_matches[joint] = value
                elif 'right' in joint:
                    right_matches[joint] = value

print("Left leg init values:")
for joint, val in sorted(left_matches.items()):
    print(f"  {joint:40s}: {val}")

print("\nRight leg init values:")
for joint, val in sorted(right_matches.items()):
    print(f"  {joint:40s}: {val}")

print("\n5. ASYMMETRY ANALYSIS:")
print("-" * 80)

issues = []
for left_joint in left_matches:
    right_joint = left_joint.replace('left', 'right')
    if right_joint in right_matches:
        if left_matches[left_joint] != right_matches[right_joint]:
            issues.append(f"{left_joint}: {left_matches[left_joint]} vs {right_joint}: {right_matches[right_joint]}")

if issues:
    print("❌ ASYMMETRIC INIT VALUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("✅ Init values are symmetric")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("""
If init values are symmetric but robot still doesn't walk symmetrically, the issue is:
  1. Contact sensor detecting differently for left vs right
  2. Reward calculation has bugs (e.g., wrong body_names pattern)
  3. USD file has physics issues (friction, collision shapes)

Next step: Run the training and check reward breakdown for left vs right foot
""")
