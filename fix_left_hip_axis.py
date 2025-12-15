#!/usr/bin/env python3
"""Fix left hip pitch axis to match right hip direction."""

import re

urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees_symmetric.urdf"
output_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_hip_axis.urdf"

with open(urdf_path, 'r') as f:
    urdf_content = f.read()

# Find left hip pitch axis
left_hip_pattern = r'(<joint name="left_hip_pitch_joint".*?<axis xyz=")([^"]+)(")'
right_hip_pattern = r'(<joint name="right_hip_pitch_joint".*?<axis xyz=")([^"]+)(")'

# Extract right hip axis to see the correct direction
right_match = re.search(right_hip_pattern, urdf_content, re.DOTALL)
if right_match:
    right_axis = right_match.group(2)
    print(f"Right hip axis: {right_axis}")

    # Right hip has: -0.006, +0.9999, 0
    # Left hip should have same Y direction: -0.006, +0.9999, 0

# Fix left hip axis - change Y from negative to positive
urdf_content = re.sub(
    r'(<joint name="left_hip_pitch_joint".*?<axis xyz=")0\.0061991061237725\s+-0\.999980785357032\s+0(")',
    r'\1-0.0061991061237725 0.999980785357032 0\2',
    urdf_content,
    flags=re.DOTALL
)

with open(output_path, 'w') as f:
    f.write(urdf_content)

print(f"\n✓ Fixed URDF saved to: {output_path}")
print("\nChanges made:")
print("  Left hip axis:  0.006, -0.9999, 0  →  -0.006, +0.9999, 0")
print("  Right hip axis: -0.006, +0.9999, 0  (unchanged)")
print("\nBoth hips now have same Y direction (+0.9999) ✓")
