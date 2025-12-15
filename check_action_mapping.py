#!/usr/bin/env python3
"""Check if action indices map correctly to joints."""

import sys
sys.path.append('/home/sncbot/IsaacLab/source/isaaclab_assets')

# Read robot config
with open('/home/sncbot/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/humanoid_snc_r5.py', 'r') as f:
    content = f.read()

print("="*80)
print("ACTION MAPPING CHECK")
print("="*80)

# Extract joint order from init_state
import re
joint_pos = re.search(r'joint_pos=\{(.*?)\}', content, re.DOTALL)
if joint_pos:
    print("\nInit state joint patterns (in order):")
    for i, line in enumerate(joint_pos.group(1).strip().split('\n')):
        line = line.strip().rstrip(',')
        if line and ':' in line:
            print(f"  {i}: {line}")

# The issue might be pattern matching order
print("\n" + "="*80)
print("PATTERN MATCHING ANALYSIS:")
print("="*80)

patterns = [
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
]

test_joints = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint"
]

print("\nPattern '.*_hip_pitch_joint' matches:")
for joint in test_joints:
    if re.match(".*_hip_pitch_joint", joint):
        print(f"  ✓ {joint}")

print("\nPattern '.*_knee_joint' matches:")
for joint in test_joints:
    if re.match(".*_knee_joint", joint):
        print(f"  ✓ {joint}")

print("\n" + "="*80)
print("ISSUE HYPOTHESIS:")
print("="*80)
print("""
When using pattern '.*_hip_pitch_joint: -0.20', BOTH left and right get -0.20

But if action is applied sequentially, and left is processed first,
then right might override it, or vice versa.

The init should be:
  left_hip_pitch_joint:  -0.20
  right_hip_pitch_joint: -0.20  (same value)

If they're getting DIFFERENT values somehow, that's the bug!
""")

print("\nCHECK INIT VALUES IN URDF:")
print("-" * 80)

import xml.etree.ElementTree as ET
tree = ET.parse('/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees_symmetric.urdf')
root = tree.getroot()

for joint in root.findall('.//joint'):
    name = joint.get('name')
    if 'hip_pitch' in name or 'knee_joint' in name:
        limit = joint.find('limit')
        if limit is not None:
            lower = limit.get('lower')
            upper = limit.get('upper')
            print(f"{name:40s}: limits [{lower}, {upper}]")

print("\n" + "="*80)
