#!/usr/bin/env python3

"""Simple test to check knee joint configuration."""

# Check joint configurations directly
print("=" * 60)
print("SNC R5 KNEE JOINT CONFIGURATION CHECK")
print("=" * 60)

# Read robot configuration
import sys
sys.path.append('/home/sncbot/IsaacLab/source/isaaclab_assets')

try:
    from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG
    
    print("✅ Robot configuration loaded successfully")
    
    # Check initial positions
    init_pos = HUMANOID_SNC_R5_CFG.init_state.joint_pos
    print(f"\nInitial Joint Positions:")
    print(f"Left knee:  {init_pos.get('left_knee_joint', 'NOT FOUND')}")
    print(f"Right knee: {init_pos.get('right_knee_joint', 'NOT FOUND')}")
    
    # Check stiffness values
    stiffness = HUMANOID_SNC_R5_CFG.actuators['body'].stiffness
    print(f"\nStiffness Values:")
    print(f"Left knee:  {stiffness.get('left_knee_joint', 'NOT FOUND')}")
    print(f"Right knee: {stiffness.get('right_knee_joint', 'NOT FOUND')}")
    
    # Check damping values
    damping = HUMANOID_SNC_R5_CFG.actuators['body'].damping
    print(f"\nDamping Values:")
    print(f"Left knee:  {damping.get('left_knee_joint', 'NOT FOUND')}")
    print(f"Right knee: {damping.get('right_knee_joint', 'NOT FOUND')}")
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS:")
    print(f"Left knee should be MORE responsive due to:")
    print(f"- Higher stiffness: {stiffness.get('left_knee_joint', 0)} vs {stiffness.get('right_knee_joint', 0)}")
    print(f"- Higher damping: {damping.get('left_knee_joint', 0)} vs {damping.get('right_knee_joint', 0)}")
    print(f"- More bent initial: {init_pos.get('left_knee_joint', 0)} vs {init_pos.get('right_knee_joint', 0)}")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error loading configuration: {e}")
    
# Test with actual training environment
print(f"\n🧪 To test knee movement during training, use:")
print(f"./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Humanoid-SNC-R5-v0 --num_envs 4 --max_iterations 50")
print(f"\nWatch the robot during training to see if both knees move equally")