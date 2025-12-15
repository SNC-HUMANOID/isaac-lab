#!/usr/bin/env python3
"""Fix SNC R5 URDF mass asymmetry."""

import xml.etree.ElementTree as ET

urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.urdf"
output_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees_symmetric.urdf"

print("Fixing mass asymmetry in SNC R5 URDF...")

tree = ET.parse(urdf_path)
root = tree.getroot()

# Fix ankle roll mass asymmetry
# Left: 0.0997, Right: 0.0978 -> Average: 0.09875
target_mass = "0.09875"

for link in root.findall('.//link'):
    name = link.get('name')
    if 'ankle_roll_link' in name:
        inertial = link.find('inertial')
        if inertial is not None:
            mass_elem = inertial.find('mass')
            if mass_elem is not None:
                old_mass = mass_elem.get('value')
                mass_elem.set('value', target_mass)
                print(f"  {name:40s}: {old_mass} -> {target_mass}")

tree.write(output_path, encoding='utf-8', xml_declaration=True)
print(f"\n✅ Fixed URDF saved to: {output_path}")
print("\nNow update humanoid_snc_r5.py to use the new URDF:")
print(f'  usd_path="{output_path.replace(".urdf", ".usd")}"')
