#!/usr/bin/env python3
"""Fix hip pitch axis direction - make left and right symmetric."""

import xml.etree.ElementTree as ET

urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees_symmetric.urdf"
output_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_axis.urdf"

print("="*80)
print("FIXING HIP AXIS DIRECTION")
print("="*80)

tree = ET.parse(urdf_path)
root = tree.getroot()

for joint in root.findall('.//joint'):
    name = joint.get('name')
    if 'hip_pitch_joint' in name:
        axis = joint.find('axis')
        if axis is not None:
            old_xyz = axis.get('xyz')
            values = [float(x) for x in old_xyz.split()]

            # Both should point in SAME direction (positive Y)
            if 'left' in name:
                # Flip left to match right
                values[0] = -values[0]  # Flip X
                values[1] = -values[1]  # Flip Y to positive
                new_xyz = f"{values[0]} {values[1]} {values[2]}"
                axis.set('xyz', new_xyz)
                print(f"{name:40s}:")
                print(f"  OLD: {old_xyz}")
                print(f"  NEW: {new_xyz}")

tree.write(output_path, encoding='utf-8', xml_declaration=True)
print(f"\n✅ Fixed URDF saved to: {output_path}")
print("\nNow update humanoid_snc_r5.py to use:")
print(f'  usd_path="{output_path.replace(".urdf", ".usd")}"')
