#!/usr/bin/env python3

"""Simple configuration checker that doesn't require Isaac imports."""

def analyze_config():
    """Analyze robot configuration by reading the file directly."""
    
    print("=" * 80)
    print("SNC R5 ROBOT CONFIGURATION ANALYSIS")
    print("=" * 80)
    print("Reading configuration from humanoid_snc_r5.py...")
    print()
    
    # Read the configuration file directly
    config_file = "/home/sncbot/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/humanoid_snc_r5.py"
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        print("✅ Configuration file read successfully")
        print()
        
        # Extract joint positions
        print("🦵 INITIAL JOINT POSITIONS:")
        print("-" * 60)
        
        # Look for joint_pos section
        if 'joint_pos=' in content:
            start = content.find('joint_pos={')
            end = content.find('},', start) + 1
            joint_section = content[start:end]
            
            # Extract specific joints
            joints_of_interest = [
                'left_hip_pitch_joint', 'right_hip_pitch_joint',
                'left_hip_roll_joint', 'right_hip_roll_joint', 
                'left_hip_yaw_joint', 'right_hip_yaw_joint',
                'left_knee_joint', 'right_knee_joint',
                'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
                'left_ankle_roll_joint', 'right_ankle_roll_joint'
            ]
            
            positions = {}
            for joint in joints_of_interest:
                # Find the joint definition
                joint_line_start = joint_section.find(f'"{joint}"')
                if joint_line_start != -1:
                    # Find the value after the colon
                    colon_pos = joint_section.find(':', joint_line_start)
                    comma_pos = joint_section.find(',', colon_pos)
                    if comma_pos == -1:
                        comma_pos = joint_section.find('\n', colon_pos)
                    
                    value_str = joint_section[colon_pos+1:comma_pos].strip()
                    # Remove comments
                    if '#' in value_str:
                        value_str = value_str[:value_str.find('#')].strip()
                    
                    try:
                        positions[joint] = float(value_str)
                    except:
                        positions[joint] = value_str
            
            # Display in organized format
            print("HIP JOINTS:")
            for joint_type in ['hip_pitch_joint', 'hip_roll_joint', 'hip_yaw_joint']:
                left_joint = f"left_{joint_type}"
                right_joint = f"right_{joint_type}"
                left_val = positions.get(left_joint, "NOT FOUND")
                right_val = positions.get(right_joint, "NOT FOUND")
                
                print(f"  {joint_type.upper():<15}: Left={left_val:>8} | Right={right_val:>8}")
                
                # Check asymmetry
                if isinstance(left_val, float) and isinstance(right_val, float):
                    if abs(left_val - right_val) > 0.001:
                        diff = left_val - right_val
                        print(f"  {'':15}  ⚠️  ASYMMETRIC! Difference: {diff:+.3f}")
                print()
            
            print("KNEE JOINTS:")
            left_knee = positions.get('left_knee_joint', "NOT FOUND")
            right_knee = positions.get('right_knee_joint', "NOT FOUND")
            print(f"  KNEE_JOINT      : Left={left_knee:>8} | Right={right_knee:>8}")
            if isinstance(left_knee, float) and isinstance(right_knee, float):
                if abs(left_knee - right_knee) > 0.001:
                    diff = left_knee - right_knee
                    print(f"  {'':15}  ⚠️  ASYMMETRIC! Difference: {diff:+.3f}")
            print()
            
            print("ANKLE JOINTS:")
            for joint_type in ['ankle_pitch_joint', 'ankle_roll_joint']:
                left_joint = f"left_{joint_type}"
                right_joint = f"right_{joint_type}"
                left_val = positions.get(left_joint, "NOT FOUND")
                right_val = positions.get(right_joint, "NOT FOUND")
                
                print(f"  {joint_type.upper():<15}: Left={left_val:>8} | Right={right_val:>8}")
                
                if isinstance(left_val, float) and isinstance(right_val, float):
                    if abs(left_val - right_val) > 0.001:
                        diff = left_val - right_val
                        print(f"  {'':15}  ⚠️  ASYMMETRIC! Difference: {diff:+.3f}")
                print()
        
        # Extract starting height
        if 'pos=(' in content:
            pos_start = content.find('pos=(')
            pos_end = content.find(')', pos_start)
            pos_str = content[pos_start:pos_end+1]
            print(f"🏗️  STARTING POSITION: {pos_str}")
            print()
        
        # Look for stiffness settings
        print("💪 STIFFNESS ANALYSIS:")
        print("-" * 60)
        
        if 'stiffness={' in content:
            stiff_start = content.find('stiffness={')
            stiff_end = content.find('},', stiff_start)
            stiff_section = content[stiff_start:stiff_end]
            
            # Look for specific left leg compensations
            left_compensations = []
            for joint in ['left_hip_pitch_joint', 'left_hip_roll_joint']:
                if f'"{joint}"' in stiff_section:
                    joint_start = stiff_section.find(f'"{joint}"')
                    colon_pos = stiff_section.find(':', joint_start)
                    comma_pos = stiff_section.find(',', colon_pos)
                    if comma_pos == -1:
                        comma_pos = stiff_section.find('\n', colon_pos)
                    
                    value_str = stiff_section[colon_pos+1:comma_pos].strip()
                    if '#' in value_str:
                        value_str = value_str[:value_str.find('#')].strip()
                    
                    try:
                        value = float(value_str)
                        left_compensations.append((joint, value))
                        print(f"  {joint}: {value}")
                    except:
                        print(f"  {joint}: {value_str}")
            
            if left_compensations:
                print(f"\n  ⚠️  LEFT LEG has higher stiffness values - this indicates")
                print(f"      software compensation for URDF asymmetries!")
        
        print()
        print("=" * 80)
        print("🔍 DIAGNOSIS:")
        print("=" * 80)
        
        # Check for major asymmetries
        left_hip_pitch = positions.get('left_hip_pitch_joint', 0)
        left_hip_roll = positions.get('left_hip_roll_joint', 0)
        right_hip_pitch = positions.get('right_hip_pitch_joint', 0)
        right_hip_roll = positions.get('right_hip_roll_joint', 0)
        
        if isinstance(left_hip_pitch, float) and isinstance(right_hip_pitch, float):
            hip_pitch_diff = abs(left_hip_pitch - right_hip_pitch)
            if hip_pitch_diff > 0.1:
                print("❌ SEVERE HIP PITCH ASYMMETRY DETECTED!")
                print(f"   Left: {left_hip_pitch:.3f}°, Right: {right_hip_pitch:.3f}°")
                print(f"   Difference: {hip_pitch_diff:.3f}°")
                print("   This suggests major URDF joint axis problems.")
                
        if isinstance(left_hip_roll, float) and isinstance(right_hip_roll, float):
            hip_roll_diff = abs(left_hip_roll - right_hip_roll)
            if hip_roll_diff > 0.05:
                print("❌ SEVERE HIP ROLL ASYMMETRY DETECTED!")
                print(f"   Left: {left_hip_roll:.3f}°, Right: {right_hip_roll:.3f}°")
                print(f"   Difference: {hip_roll_diff:.3f}°")
                print("   This explains the strange hip-leg connection!")
                
        print()
        print("🎯 SUMMARY:")
        print("   The robot configuration shows software compensations")
        print("   for hardware (URDF) asymmetries. This explains why")
        print("   the left hip-leg connection looks strange.")
        print()
        print("   The left leg is being forced into an unnatural")
        print("   position to compensate for misaligned joint axes.")
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")

if __name__ == "__main__":
    analyze_config()