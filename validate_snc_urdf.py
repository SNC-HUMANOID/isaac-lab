#!/usr/bin/env python3
"""Validate SNC R5 URDF/USD for asymmetry issues."""

import xml.etree.ElementTree as ET
import re

urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.urdf"

print("="*80)
print("SNC R5 URDF VALIDATION")
print("="*80)

tree = ET.parse(urdf_path)
root = tree.getroot()

# 1. Check joints
print("\n1. JOINT NAMES CHECK:")
print("-" * 80)
joints = {}
for joint in root.findall('.//joint'):
    name = joint.get('name')
    jtype = joint.get('type')
    joints[name] = jtype

left_joints = {k: v for k, v in joints.items() if 'left' in k and any(x in k for x in ['hip', 'knee', 'ankle'])}
right_joints = {k: v for k, v in joints.items() if 'right' in k and any(x in k for x in ['hip', 'knee', 'ankle'])}

print(f"Left leg joints: {len(left_joints)}")
for name in sorted(left_joints.keys()):
    print(f"  ✓ {name:30s} ({left_joints[name]})")

print(f"\nRight leg joints: {len(right_joints)}")
for name in sorted(right_joints.keys()):
    print(f"  ✓ {name:30s} ({right_joints[name]})")

if len(left_joints) != len(right_joints):
    print(f"\n❌ ASYMMETRY: Left has {len(left_joints)} joints, Right has {len(right_joints)}")
else:
    print(f"\n✅ Joint count symmetric: {len(left_joints)} joints per leg")

# 2. Check links and masses
print("\n\n2. LINK MASS CHECK:")
print("-" * 80)
links = {}
for link in root.findall('.//link'):
    name = link.get('name')
    inertial = link.find('inertial')
    if inertial is not None:
        mass_elem = inertial.find('mass')
        mass = float(mass_elem.get('value')) if mass_elem is not None else 0.0
        links[name] = mass
    else:
        links[name] = 0.0

left_links = {k: v for k, v in links.items() if 'left' in k and any(x in k for x in ['hip', 'knee', 'ankle', 'thigh', 'shank'])}
right_links = {k: v for k, v in links.items() if 'right' in k and any(x in k for x in ['hip', 'knee', 'ankle', 'thigh', 'shank'])}

print("Left leg link masses:")
total_left = 0
for name in sorted(left_links.keys()):
    mass = left_links[name]
    total_left += mass
    print(f"  {name:40s}: {mass:8.4f} kg")

print(f"\nRight leg link masses:")
total_right = 0
for name in sorted(right_links.keys()):
    mass = right_links[name]
    total_right += mass
    print(f"  {name:40s}: {mass:8.4f} kg")

print(f"\nTotal left leg mass:  {total_left:.4f} kg")
print(f"Total right leg mass: {total_right:.4f} kg")
diff = abs(total_left - total_right)
if diff > 0.001:
    print(f"❌ MASS ASYMMETRY: {diff:.4f} kg difference!")
else:
    print(f"✅ Mass symmetric (diff: {diff:.6f} kg)")

# 3. Check joint limits
print("\n\n3. JOINT LIMITS CHECK:")
print("-" * 80)
for joint in root.findall('.//joint'):
    name = joint.get('name')
    if 'left' in name and any(x in name for x in ['hip', 'knee', 'ankle']):
        limit = joint.find('limit')
        if limit is not None:
            lower = float(limit.get('lower', '0'))
            upper = float(limit.get('upper', '0'))
            print(f"  {name:40s}: [{lower:7.3f}, {upper:7.3f}]")

# 4. Check for contact sensors
print("\n\n4. CONTACT/COLLISION CHECK:")
print("-" * 80)
collisions = root.findall('.//collision')
print(f"Total collision elements: {len(collisions)}")

ankle_collisions = [c for c in collisions if 'ankle' in c.getparent().get('name', '').lower()]
print(f"Ankle collision elements: {len(ankle_collisions)}")

left_ankle = [c for c in ankle_collisions if 'left' in c.getparent().get('name', '').lower()]
right_ankle = [c for c in ankle_collisions if 'right' in c.getparent().get('name', '').lower()]

print(f"  Left ankle collisions: {len(left_ankle)}")
print(f"  Right ankle collisions: {len(right_ankle)}")

if len(left_ankle) != len(right_ankle):
    print(f"❌ COLLISION ASYMMETRY: Left {len(left_ankle)}, Right {len(right_ankle)}")
else:
    print(f"✅ Collision symmetric")

# 5. Summary
print("\n\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

issues = []
if len(left_joints) != len(right_joints):
    issues.append("Joint count asymmetry")
if abs(total_left - total_right) > 0.001:
    issues.append(f"Mass asymmetry ({diff:.4f} kg)")
if len(left_ankle) != len(right_ankle):
    issues.append("Collision asymmetry")

if issues:
    print("\n❌ ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nThese asymmetries may cause the left leg to behave differently!")
else:
    print("\n✅ URDF appears symmetric")
    print("\nIf robot still doesn't walk properly, check:")
    print("  1. USD conversion from URDF")
    print("  2. Physics simulation parameters")
    print("  3. Reward function implementation")

print("\n" + "="*80)
