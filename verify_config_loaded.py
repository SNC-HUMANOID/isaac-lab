#!/usr/bin/env python3
"""Verify that our config changes are actually loaded."""

import sys
import importlib

print("="*60)
print("CONFIG VERIFICATION")
print("="*60)

# Import the config
try:
    cfg_module = importlib.import_module(
        "isaaclab_tasks.manager_based.locomotion.velocity.config.humanoid_snc_r5.rough_env_cfg"
    )
    print("\n✓ Successfully imported rough_env_cfg")

    # Create config instance
    env_cfg = cfg_module.HumanoidSNCR5RoughEnvCfg()

    print(f"\n📋 Action joint_pos config:")
    print(f"   Type: {type(env_cfg.actions.joint_pos)}")

    if hasattr(env_cfg.actions.joint_pos, 'scale'):
        scale = env_cfg.actions.joint_pos.scale
        print(f"   Scale type: {type(scale)}")

        if isinstance(scale, dict):
            print(f"   Scale is DICTIONARY ✓")
            print(f"\n   Key joints:")
            if "left_hip_pitch_joint" in scale:
                print(f"      left_hip_pitch_joint:  {scale['left_hip_pitch_joint']:+.1f} {'← INVERTED!' if scale['left_hip_pitch_joint'] < 0 else ''}")
            if "right_hip_pitch_joint" in scale:
                print(f"      right_hip_pitch_joint: {scale['right_hip_pitch_joint']:+.1f}")
        else:
            print(f"   ❌ Scale is NOT dictionary: {scale}")
    else:
        print(f"   ❌ No scale attribute found")

    print(f"\n📋 Initial state:")
    if hasattr(env_cfg.scene.robot.init_state, 'joint_pos'):
        joint_pos = env_cfg.scene.robot.init_state.joint_pos
        print(f"   Type: {type(joint_pos)}")
        print(f"   Keys: {list(joint_pos.keys())}")

        for key in ["left_hip_pitch_joint", "right_hip_pitch_joint", ".*_hip_pitch_joint"]:
            if key in joint_pos:
                print(f"      {key}: {joint_pos[key]:+.2f}")

    print(f"\n{'='*60}")
    print("✓ Config loaded successfully")
    print("="*60)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
