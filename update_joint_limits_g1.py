#!/usr/bin/env python3
"""Update joint limits in Humanoid_10102025 URDF to match G1 humanoid."""

import xml.etree.ElementTree as ET

# G1 humanoid joint limits (in radians)
G1_JOINT_LIMITS = {
    # Hip joints
    "left_hip_pitch_joint": {"lower": -0.43, "upper": 2.79, "effort": 200, "velocity": 30},
    "right_hip_pitch_joint": {"lower": -0.43, "upper": 2.79, "effort": 200, "velocity": 30},
    "left_hip_roll_joint": {"lower": -0.43, "upper": 0.43, "effort": 200, "velocity": 30},
    "right_hip_roll_joint": {"lower": -0.43, "upper": 0.43, "effort": 200, "velocity": 30},
    "left_hip_yaw_joint": {"lower": -0.43, "upper": 0.43, "effort": 200, "velocity": 30},
    "right_hip_yaw_joint": {"lower": -0.43, "upper": 0.43, "effort": 200, "velocity": 30},
    
    # Knee joints
    "left_knee_joint": {"lower": -0.26, "upper": 2.05, "effort": 300, "velocity": 30},
    "right_knee_joint": {"lower": -0.26, "upper": 2.05, "effort": 300, "velocity": 30},
    
    # Ankle joints
    "left_ankle_pitch_joint": {"lower": -0.87, "upper": 0.52, "effort": 45, "velocity": 30},
    "right_ankle_pitch_joint": {"lower": -0.87, "upper": 0.52, "effort": 45, "velocity": 30},
    "left_ankle_roll_joint": {"lower": -0.26, "upper": 0.26, "effort": 45, "velocity": 30},
    "right_ankle_roll_joint": {"lower": -0.26, "upper": 0.26, "effort": 45, "velocity": 30},
    
    # Shoulder joints
    "left_shoulder_pitch_joint": {"lower": -2.87, "upper": 2.87, "effort": 45, "velocity": 30},
    "right_shoulder_pitch_joint": {"lower": -2.87, "upper": 2.87, "effort": 45, "velocity": 30},
    "left_shoulder_roll_joint": {"lower": -1.57, "upper": 3.11, "effort": 45, "velocity": 30},
    "right_shoulder_roll_joint": {"lower": -3.11, "upper": 1.57, "effort": 45, "velocity": 30},
    "left_shoulder_yaw_joint": {"lower": -3.01, "upper": 1.32, "effort": 45, "velocity": 30},
    "right_shoulder_yaw_joint": {"lower": -1.32, "upper": 3.01, "effort": 45, "velocity": 30},
    
    # Elbow joints  
    "left_elbow_joint": {"lower": -2.23, "upper": 0.09, "effort": 45, "velocity": 30},
    "right_elbow_joint": {"lower": -2.23, "upper": 0.09, "effort": 45, "velocity": 30},
    
    # Pelvis joint (waist)
    "pelvis_joint": {"lower": -2.79, "upper": 2.79, "effort": 200, "velocity": 30},
}

def update_urdf(input_file, output_file):
    """Update joint limits in URDF file."""
    
    # Parse URDF
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    updated_count = 0
    
    # Find all joints
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        # Only update revolute joints
        if joint_type == 'revolute' and joint_name in G1_JOINT_LIMITS:
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                limits = G1_JOINT_LIMITS[joint_name]
                
                # Update limits
                limit_elem.set('lower', str(limits['lower']))
                limit_elem.set('upper', str(limits['upper']))
                limit_elem.set('effort', str(limits['effort']))
                limit_elem.set('velocity', str(limits['velocity']))
                
                print(f"✓ Updated {joint_name}:")
                print(f"  Range: [{limits['lower']:.2f}, {limits['upper']:.2f}] rad")
                print(f"  Effort: {limits['effort']} Nm, Velocity: {limits['velocity']} rad/s")
                
                updated_count += 1
    
    # Save updated URDF
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"\n✅ Updated {updated_count} joints")
    print(f"📝 Saved to: {output_file}")

if __name__ == "__main__":
    input_urdf = "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/urdf/Humanoid_10102025.urdf"
    output_urdf = "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/urdf/Humanoid_10102025_g1_limits.urdf"
    
    print("=" * 60)
    print("Updating Joint Limits to G1 Humanoid Spec")
    print("=" * 60)
    
    update_urdf(input_urdf, output_urdf)
    
    print("\n" + "=" * 60)
    print("Next step: Convert to USD")
    print("=" * 60)
    print("\nRun this command:")
    print("./isaaclab.sh -p scripts/tools/convert_urdf.py \\")
    print(f"  {output_urdf} \\")
    print("  /home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025_g1.usd \\")
    print("  --merge-joints")

