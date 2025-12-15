#!/usr/bin/env python3

"""Inspect SNC R5 robot configuration without simulation."""

import sys
import os

# Add IsaacLab to path
sys.path.append('/home/sncbot/IsaacLab/source/isaaclab_assets')

def inspect_robot_config():
    """Inspect robot configuration to show joint settings."""
    
    print("=" * 80)
    print("SNC R5 ROBOT CONFIGURATION INSPECTOR")
    print("=" * 80)
    print("Examining joint positions and compensations...")
    print()
    
    try:
        from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG
        
        print("✅ Robot configuration loaded successfully")
        print()
        
        # Get configuration details
        init_pos = HUMANOID_SNC_R5_CFG.init_state.joint_pos
        stiffness = HUMANOID_SNC_R5_CFG.actuators['body'].stiffness
        damping = HUMANOID_SNC_R5_CFG.actuators['body'].damping
        starting_height = HUMANOID_SNC_R5_CFG.init_state.pos[2]
        
        print(f"🏗️  ROBOT SPAWN CONFIGURATION:")
        print(f"   Starting height: {starting_height:.3f}m")
        print(f"   USD file: {HUMANOID_SNC_R5_CFG.spawn.usd_path}")
        print()
        
        print("🦵 LEG JOINT ANALYSIS:")
        print("-" * 60)
        
        # Hip joints
        print("HIP JOINTS:")
        hip_joints = ['hip_pitch_joint', 'hip_roll_joint', 'hip_yaw_joint']
        for joint in hip_joints:
            left_joint = f"left_{joint}"
            right_joint = f"right_{joint}"
            
            left_pos = init_pos.get(left_joint, 0.0)
            right_pos = init_pos.get(right_joint, 0.0)
            left_stiff = stiffness.get(left_joint, "DEFAULT")
            right_stiff = stiffness.get(right_joint, "DEFAULT")
            left_damp = damping.get(left_joint, "DEFAULT")
            right_damp = damping.get(right_joint, "DEFAULT")
            
            print(f"  {joint.upper():<15}: Left={left_pos:6.3f}° | Right={right_pos:6.3f}°")
            print(f"  {'':15}  Stiff: {left_stiff:<6} | {right_stiff:<6}")
            print(f"  {'':15}  Damp:  {left_damp:<6} | {right_damp:<6}")
            
            # Check for asymmetry
            if left_pos != right_pos:
                print(f"  {'':15}  ⚠️  ASYMMETRIC INITIAL POSITION!")
            if left_stiff != right_stiff:
                print(f"  {'':15}  ⚠️  ASYMMETRIC STIFFNESS!")
            print()
        
        # Knee joints
        print("KNEE JOINTS:")
        left_knee_pos = init_pos.get('left_knee_joint', 0.0)
        right_knee_pos = init_pos.get('right_knee_joint', 0.0)
        left_knee_stiff = stiffness.get('left_knee_joint', "DEFAULT")
        right_knee_stiff = stiffness.get('right_knee_joint', "DEFAULT")
        left_knee_damp = damping.get('left_knee_joint', "DEFAULT")
        right_knee_damp = damping.get('right_knee_joint', "DEFAULT")
        
        print(f"  KNEE_JOINT      : Left={left_knee_pos:6.3f}° | Right={right_knee_pos:6.3f}°")
        print(f"  {'':15}  Stiff: {left_knee_stiff:<6} | {right_knee_stiff:<6}")
        print(f"  {'':15}  Damp:  {left_knee_damp:<6} | {right_knee_damp:<6}")
        
        if left_knee_pos != right_knee_pos:
            print(f"  {'':15}  ⚠️  ASYMMETRIC INITIAL POSITION!")
        if left_knee_stiff != right_knee_stiff:
            print(f"  {'':15}  ⚠️  ASYMMETRIC STIFFNESS!")
        print()
        
        # Ankle joints
        print("ANKLE JOINTS:")
        ankle_joints = ['ankle_pitch_joint', 'ankle_roll_joint']
        for joint in ankle_joints:
            left_joint = f"left_{joint}"
            right_joint = f"right_{joint}"
            
            left_pos = init_pos.get(left_joint, 0.0)
            right_pos = init_pos.get(right_joint, 0.0)
            left_stiff = stiffness.get(left_joint, "DEFAULT")
            right_stiff = stiffness.get(right_joint, "DEFAULT")
            left_damp = damping.get(left_joint, "DEFAULT")
            right_damp = damping.get(right_joint, "DEFAULT")
            
            print(f"  {joint.upper():<15}: Left={left_pos:6.3f}° | Right={right_pos:6.3f}°")
            print(f"  {'':15}  Stiff: {left_stiff:<6} | {right_stiff:<6}")
            print(f"  {'':15}  Damp:  {left_damp:<6} | {right_damp:<6}")
            
            if left_pos != right_pos:
                print(f"  {'':15}  ⚠️  ASYMMETRIC INITIAL POSITION!")
            if left_stiff != right_stiff:
                print(f"  {'':15}  ⚠️  ASYMMETRIC STIFFNESS!")
            print()
        
        print("=" * 80)
        print("🔍 COMPENSATION ANALYSIS:")
        print("=" * 80)
        
        # Analyze left leg compensations
        left_hip_pitch = init_pos.get('left_hip_pitch_joint', 0.0)
        left_hip_roll = init_pos.get('left_hip_roll_joint', 0.0)
        left_hip_yaw = init_pos.get('left_hip_yaw_joint', 0.0)
        
        print(f"LEFT LEG COMPENSATIONS:")
        print(f"  Hip Pitch: {left_hip_pitch:+.3f}° (forward/backward lean)")
        print(f"  Hip Roll:  {left_hip_roll:+.3f}° (side lean)")
        print(f"  Hip Yaw:   {left_hip_yaw:+.3f}° (rotation)")
        
        if abs(left_hip_pitch) > 0.05:
            print(f"  ⚠️  Large hip pitch compensation! May indicate URDF axis misalignment")
        if abs(left_hip_roll) > 0.05:
            print(f"  ⚠️  Large hip roll compensation! May indicate URDF axis misalignment")
        
        print()
        print(f"LEFT LEG STIFFNESS BOOST:")
        left_hp_stiff = stiffness.get('left_hip_pitch_joint', 0)
        right_hp_stiff = stiffness.get('right_hip_pitch_joint', 0)
        left_hr_stiff = stiffness.get('left_hip_roll_joint', 0)
        right_hr_stiff = stiffness.get('right_hip_roll_joint', 0)
        
        print(f"  Hip Pitch: {left_hp_stiff} vs {right_hp_stiff} (boost: {left_hp_stiff - right_hp_stiff:+})")
        print(f"  Hip Roll:  {left_hr_stiff} vs {right_hr_stiff} (boost: {left_hr_stiff - right_hr_stiff:+})")
        
        print()
        print("=" * 80)
        print("📝 CONCLUSION:")
        print("=" * 80)
        
        if abs(left_hip_pitch) > 0.1 or abs(left_hip_roll) > 0.05:
            print("❌ SEVERE URDF ASYMMETRY DETECTED!")
            print("   The left leg requires large compensations, indicating:")
            print("   • Hip joint axes are not symmetric in the URDF file")
            print("   • Hardware geometry issues in the original model")
            print("   • Software compensation is being used instead of hardware fix")
            print()
            print("🛠️  RECOMMENDATIONS:")
            print("   1. Fix URDF joint axes to be perfectly symmetric")
            print("   2. Or use a different robot model without asymmetries")
            print("   3. Current software compensation may cause walking instability")
        else:
            print("✅ Robot configuration looks symmetric!")
            print("   All legs should behave similarly during training.")
        
        print()
        print("🎬 To visualize the robot with these settings, use:")
        print("   ./isaaclab.sh -p scripts/tools/check_instanceable.py --usd_path")
        print("   OR try a Manager-Based Environment training to see if it works better.")
        
    except Exception as e:
        print(f"❌ Error loading robot configuration: {e}")
        print("Make sure IsaacLab is properly set up and paths are correct.")

if __name__ == "__main__":
    inspect_robot_config()