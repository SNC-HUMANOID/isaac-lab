#!/usr/bin/env python3
"""
G1 Robot Kinematics Export Script
Based on Isaac Lab training configuration
"""

import numpy as np

# G1 Robot Joint Information (extracted from training config)
G1_JOINTS = {
    # Leg joints (from config analysis)
    'left_hip_yaw_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'legs'},
    'left_hip_roll_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'legs'},
    'left_hip_pitch_joint': {'initial_pos': -0.2, 'type': 'revolute', 'group': 'legs'},
    'left_knee_joint': {'initial_pos': 0.42, 'type': 'revolute', 'group': 'legs'},
    'left_ankle_pitch_joint': {'initial_pos': -0.23, 'type': 'revolute', 'group': 'feet'},
    'left_ankle_roll_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'feet'},
    
    'right_hip_yaw_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'legs'},
    'right_hip_roll_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'legs'},
    'right_hip_pitch_joint': {'initial_pos': -0.2, 'type': 'revolute', 'group': 'legs'},
    'right_knee_joint': {'initial_pos': 0.42, 'type': 'revolute', 'group': 'legs'},
    'right_ankle_pitch_joint': {'initial_pos': -0.23, 'type': 'revolute', 'group': 'feet'},
    'right_ankle_roll_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'feet'},
    
    # Torso
    'torso_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'legs'},
    
    # Arm joints
    'left_shoulder_pitch_joint': {'initial_pos': 0.35, 'type': 'revolute', 'group': 'arms'},
    'left_shoulder_roll_joint': {'initial_pos': 0.16, 'type': 'revolute', 'group': 'arms'},
    'left_shoulder_yaw_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'left_elbow_pitch_joint': {'initial_pos': 0.87, 'type': 'revolute', 'group': 'arms'},
    'left_elbow_roll_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    
    'right_shoulder_pitch_joint': {'initial_pos': 0.35, 'type': 'revolute', 'group': 'arms'},
    'right_shoulder_roll_joint': {'initial_pos': -0.16, 'type': 'revolute', 'group': 'arms'},
    'right_shoulder_yaw_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'right_elbow_pitch_joint': {'initial_pos': 0.87, 'type': 'revolute', 'group': 'arms'},
    'right_elbow_roll_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    
    # Hand/finger joints
    'left_zero_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'left_one_joint': {'initial_pos': 1.0, 'type': 'revolute', 'group': 'arms'},
    'left_two_joint': {'initial_pos': 0.52, 'type': 'revolute', 'group': 'arms'},
    'left_three_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'left_four_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'left_five_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'left_six_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    
    'right_zero_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'right_one_joint': {'initial_pos': -1.0, 'type': 'revolute', 'group': 'arms'},
    'right_two_joint': {'initial_pos': -0.52, 'type': 'revolute', 'group': 'arms'},
    'right_three_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'right_four_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'right_five_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
    'right_six_joint': {'initial_pos': 0.0, 'type': 'revolute', 'group': 'arms'},
}

# Actuator groups information (from training config)
ACTUATOR_GROUPS = {
    'legs': {
        'joints': ['*_hip_yaw_joint', '*_hip_roll_joint', '*_hip_pitch_joint', '*_knee_joint', 'torso_joint'],
        'effort_limit': 300,  # Nm
        'velocity_limit': 100.0,  # rad/s
        'stiffness': {'hip_yaw': 150.0, 'hip_roll': 150.0, 'hip_pitch': 200.0, 'knee': 200.0, 'torso': 200.0},
        'damping': {'hip_yaw': 5.0, 'hip_roll': 5.0, 'hip_pitch': 5.0, 'knee': 5.0, 'torso': 5.0}
    },
    'feet': {
        'joints': ['*_ankle_pitch_joint', '*_ankle_roll_joint'],
        'effort_limit': 20,  # Nm
        'stiffness': 20.0,
        'damping': 2.0
    },
    'arms': {
        'joints': ['*_shoulder_*_joint', '*_elbow_*_joint', '*_*_joint'],
        'effort_limit': 300,  # Nm
        'velocity_limit': 100.0,  # rad/s
        'stiffness': 40.0,
        'damping': 10.0
    }
}

# Robot physical properties
ROBOT_PROPERTIES = {
    'initial_position': [0.0, 0.0, 0.74],  # [x, y, z] in meters
    'initial_orientation': [1.0, 0.0, 0.0, 0.0],  # quaternion [w, x, y, z]
    'mass_estimate': 35.0,  # kg (typical for G1)
    'height': 1.32,  # meters (typical G1 height)
    'usd_model_url': 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd'
}

def get_joint_names():
    """Get all joint names"""
    return list(G1_JOINTS.keys())

def get_joint_names_by_group(group_name):
    """Get joint names by actuator group"""
    return [name for name, info in G1_JOINTS.items() if info['group'] == group_name]

def get_initial_joint_positions():
    """Get initial joint positions as numpy array"""
    return np.array([G1_JOINTS[joint]['initial_pos'] for joint in get_joint_names()])

def get_joint_limits():
    """Get approximate joint limits (typical for humanoid robots)"""
    joint_names = get_joint_names()
    
    # Default conservative limits for humanoid robots
    limits = {
        'lower': np.array([-np.pi] * len(joint_names)),
        'upper': np.array([np.pi] * len(joint_names))
    }
    
    # More specific limits for certain joints
    for i, name in enumerate(joint_names):
        if 'knee' in name:
            limits['lower'][i] = -0.1  # Knees typically don't bend backwards much
            limits['upper'][i] = 2.5   # About 140 degrees forward
        elif 'ankle' in name:
            limits['lower'][i] = -0.5  # About 30 degrees
            limits['upper'][i] = 0.5   # About 30 degrees
        elif 'elbow' in name and 'pitch' in name:
            limits['lower'][i] = -0.1  # Slight extension
            limits['upper'][i] = 2.5   # About 140 degrees flexion
    
    return limits

def print_robot_summary():
    """Print comprehensive robot information"""
    print("🤖 G1 Robot Kinematics Summary")
    print("=" * 60)
    
    print(f"📍 Physical Properties:")
    print(f"   • Initial position: {ROBOT_PROPERTIES['initial_position']} m")
    print(f"   • Initial height: {ROBOT_PROPERTIES['initial_position'][2]} m")
    print(f"   • Estimated mass: {ROBOT_PROPERTIES['mass_estimate']} kg")
    print(f"   • Total joints: {len(G1_JOINTS)}")
    
    print(f"\n🦵 Joint Groups:")
    for group in ['legs', 'feet', 'arms']:
        joints = get_joint_names_by_group(group)
        print(f"   • {group.upper()}: {len(joints)} joints")
        for joint in joints[:3]:  # Show first 3
            pos = G1_JOINTS[joint]['initial_pos']
            print(f"     - {joint}: {pos:.3f} rad ({np.degrees(pos):.1f}°)")
        if len(joints) > 3:
            print(f"     - ... and {len(joints)-3} more")
    
    print(f"\n⚙️  Actuator Specifications:")
    for group_name, group_info in ACTUATOR_GROUPS.items():
        print(f"   🔧 {group_name.upper()}:")
        print(f"      - Max effort: {group_info['effort_limit']} Nm")
        if 'velocity_limit' in group_info:
            print(f"      - Max velocity: {group_info['velocity_limit']} rad/s")
        if isinstance(group_info.get('stiffness'), dict):
            print(f"      - Stiffness range: {min(group_info['stiffness'].values())}-{max(group_info['stiffness'].values())} Nm/rad")
        elif 'stiffness' in group_info:
            print(f"      - Stiffness: {group_info['stiffness']} Nm/rad")

def calculate_simple_forward_kinematics():
    """
    Simple forward kinematics calculation for demonstration
    (This is a simplified version - real FK would need proper DH parameters)
    """
    print("\n🧮 Simple Forward Kinematics Calculation:")
    print("=" * 50)
    
    # Get initial positions
    joint_positions = get_initial_joint_positions()
    joint_names = get_joint_names()
    
    # Calculate approximate foot positions (simplified)
    # This is just for demonstration - real FK needs proper kinematic chain
    
    # Leg length segments (approximate for G1)
    upper_leg_length = 0.35  # meters
    lower_leg_length = 0.35  # meters
    
    print("📐 Leg Kinematics (Simplified):")
    
    for side in ['left', 'right']:
        try:
            # Find relevant joint indices
            hip_pitch_idx = joint_names.index(f'{side}_hip_pitch_joint')
            knee_idx = joint_names.index(f'{side}_knee_joint')
            ankle_pitch_idx = joint_names.index(f'{side}_ankle_pitch_joint')
            
            # Get joint angles
            hip_pitch = joint_positions[hip_pitch_idx]
            knee_angle = joint_positions[knee_idx]
            ankle_pitch = joint_positions[ankle_pitch_idx]
            
            # Simple 2D leg kinematics in sagittal plane
            # Hip to knee
            knee_x = upper_leg_length * np.sin(hip_pitch)
            knee_z = -upper_leg_length * np.cos(hip_pitch)
            
            # Knee to ankle
            ankle_x = knee_x + lower_leg_length * np.sin(hip_pitch + knee_angle)
            ankle_z = knee_z - lower_leg_length * np.cos(hip_pitch + knee_angle)
            
            # Relative to robot base
            base_height = ROBOT_PROPERTIES['initial_position'][2]
            foot_height = base_height + ankle_z
            
            print(f"   {side.upper()} leg:")
            print(f"     • Hip pitch: {hip_pitch:.3f} rad ({np.degrees(hip_pitch):.1f}°)")
            print(f"     • Knee angle: {knee_angle:.3f} rad ({np.degrees(knee_angle):.1f}°)")
            print(f"     • Ankle pitch: {ankle_pitch:.3f} rad ({np.degrees(ankle_pitch):.1f}°)")
            print(f"     • Foot position: x={ankle_x:.3f}m, z={foot_height:.3f}m")
            
        except ValueError as e:
            print(f"   Could not find joints for {side} leg: {e}")

def export_for_external_libraries():
    """Generate data for external kinematics libraries"""
    print("\n📤 Export for External Libraries:")
    print("=" * 50)
    
    # For PyBullet
    print("🐍 PyBullet format:")
    print("   joint_names = [")
    for name in get_joint_names():
        print(f"       '{name}',")
    print("   ]")
    print(f"   initial_positions = {get_initial_joint_positions().tolist()}")
    
    # For Pinocchio/URDF
    print("\n🔧 URDF joint information:")
    print("   <!-- Copy this into your URDF file -->")
    limits = get_joint_limits()
    for i, name in enumerate(get_joint_names()):
        initial_pos = G1_JOINTS[name]['initial_pos']
        lower = limits['lower'][i]
        upper = limits['upper'][i]
        print(f"   <joint name='{name}' type='revolute'>")
        print(f"     <limit lower='{lower:.3f}' upper='{upper:.3f}' effort='100' velocity='10'/>")
        print(f"     <!-- Initial position: {initial_pos:.3f} rad -->")
        print(f"   </joint>")
        if i >= 2:  # Show only first 3 for brevity
            print(f"   <!-- ... and {len(get_joint_names())-3} more joints -->")
            break

def test_with_external_libraries():
    """Test integration with external libraries if available"""
    print("\n🧪 Testing External Library Integration:")
    print("=" * 50)
    
    # Test numpy operations
    print("✅ NumPy operations:")
    positions = get_initial_joint_positions()
    print(f"   • Mean joint position: {np.mean(positions):.3f} rad")
    print(f"   • Joint position range: {np.min(positions):.3f} to {np.max(positions):.3f} rad")
    
    # Test PyBullet if available
    try:
        import pybullet as p
        print("✅ PyBullet available:")
        print("   • Ready for physics simulation")
        print("   • Can load robot with: p.loadURDF('g1_robot.urdf')")
    except ImportError:
        print("❌ PyBullet not available (install with: pip install pybullet)")
    
    # Test Pinocchio if available
    try:
        import pinocchio as pin
        print("✅ Pinocchio available:")
        print("   • Ready for advanced kinematics")
        print("   • Can load with: pin.buildModelFromUrdf('g1_robot.urdf')")
    except ImportError:
        print("❌ Pinocchio not available (install with: pip install pin)")
    
    # Test matplotlib for visualization
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib available:")
        print("   • Ready for joint angle visualization")
    except ImportError:
        print("❌ Matplotlib not available (install with: pip install matplotlib)")

if __name__ == "__main__":
    print_robot_summary()
    calculate_simple_forward_kinematics()
    export_for_external_libraries()
    test_with_external_libraries()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Convert USD model to URDF using Isaac Lab tools")
    print("2. Use this joint information with PyBullet/Pinocchio")
    print("3. Implement full forward/inverse kinematics")
    print("4. Test with the trained RL policy")
    print("\n💡 Commands to try:")
    print("   # Convert USD to URDF")
    print("   ./isaaclab.sh -p scripts/tools/convert_urdf.py --input_path <usd_file>")
    print("   # Run trained policy")
    print("   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\")
    print("     --task Isaac-Velocity-Flat-G1-v0 --load_run logs/rsl_rl/g1_flat/2025-07-29_15-27-26/")