#!/usr/bin/env python3
"""แยกข้อมูล robot จาก CSV เพื่อปรับปรุง RL training"""

import pandas as pd
import json

def extract_robot_specs():
    """แยกข้อมูลสำคัญจาก Humanoid_SNC.csv"""
    
    # อ่าน CSV file
    csv_path = "/home/sncbot/Downloads/HM+lage/Humaniod_SNC_ROS_URDF/Humanoid_SNC/urdf/Humanoid_SNC.csv"
    df = pd.read_csv(csv_path)
    
    # แยกข้อมูล joints
    joints_data = {}
    
    for _, row in df.iterrows():
        if pd.notna(row['Joint Name']) and row['Joint Name'] != '':
            joint_name = row['Joint Name']
            
            joints_data[joint_name] = {
                'type': row['Joint Type'],
                'effort_limit': row['Limit Effort'] if pd.notna(row['Limit Effort']) else 100.0,
                'velocity_limit': row['Limit Velocity'] if pd.notna(row['Limit Velocity']) else 10.0,
                'position_lower': row['Limit Lower'] if pd.notna(row['Limit Lower']) else -3.14,
                'position_upper': row['Limit Upper'] if pd.notna(row['Limit Upper']) else 3.14,
                'mass': row['Mass'] if pd.notna(row['Mass']) else 1.0,
                'inertia_xx': row['Moment Ixx'] if pd.notna(row['Moment Ixx']) else 0.01,
                'inertia_yy': row['Moment Iyy'] if pd.notna(row['Moment Iyy']) else 0.01,
                'inertia_zz': row['Moment Izz'] if pd.notna(row['Moment Izz']) else 0.01,
            }
    
    # สรุปข้อมูล
    summary = {
        'total_joints': len(joints_data),
        'total_mass': df['Mass'].sum(),
        'joint_effort_limits': [joints_data[j]['effort_limit'] for j in joints_data],
        'joint_velocity_limits': [joints_data[j]['velocity_limit'] for j in joints_data],
        'joint_position_ranges': [(joints_data[j]['position_lower'], joints_data[j]['position_upper']) for j in joints_data]
    }
    
    print("🤖 Humanoid SNC Robot Specifications")
    print("=" * 50)
    print(f"📊 Total joints: {summary['total_joints']}")
    print(f"⚖️  Total mass: {summary['total_mass']:.2f} kg")
    print(f"💪 Effort limits: {min(summary['joint_effort_limits']):.1f} - {max(summary['joint_effort_limits']):.1f} Nm")
    print(f"🏃 Velocity limits: {min(summary['joint_velocity_limits']):.1f} - {max(summary['joint_velocity_limits']):.1f} rad/s")
    
    print("\n🔧 Joint Details:")
    for joint_name, data in joints_data.items():
        print(f"  {joint_name:25s}: effort={data['effort_limit']:5.1f}Nm, vel={data['velocity_limit']:4.1f}rad/s, range=[{data['position_lower']:6.3f}, {data['position_upper']:6.3f}]")
    
    # บันทึกเป็น JSON
    output_path = "/home/sncbot/IsaacLab/humanoid_snc_scripts/05_improved_config/robot_specs.json"
    with open(output_path, 'w') as f:
        json.dump({'joints': joints_data, 'summary': summary}, f, indent=2)
    
    print(f"\n💾 Saved specs to: {output_path}")
    
    return joints_data, summary

if __name__ == "__main__":
    try:
        joints, summary = extract_robot_specs()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("📋 Available data in original folder:")
        print("  - URDF file (already converted to USD)")  
        print("  - STL meshes (already integrated)")
        print("  - Joint limits and effort data")
        print("  - Mass and inertia properties")
        print("\n💡 This data can improve RL training by:")
        print("  - Setting realistic joint limits")
        print("  - Using proper effort/velocity limits")  
        print("  - Adjusting reward weights by mass")
        print("  - Better PD controller parameters")