#!/usr/bin/env python3
"""Test script to verify Humanoid_abs configuration is properly set up."""

import sys

def test_imports():
    """Test that all necessary modules can be imported."""
    print("Testing imports...")
    
    try:
        from isaaclab_assets import HUMANOID_ABS_CFG
        print("✓ HUMANOID_ABS_CFG imported successfully")
    except Exception as e:
        print(f"✗ Failed to import HUMANOID_ABS_CFG: {e}")
        return False
    
    try:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.humanoid_abs import flat_env_cfg
        print("✓ flat_env_cfg module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import flat_env_cfg: {e}")
        return False
    
    try:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.humanoid_abs.agents import rsl_rl_ppo_cfg
        print("✓ rsl_rl_ppo_cfg module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import rsl_rl_ppo_cfg: {e}")
        return False
    
    return True

def test_gym_registration():
    """Test that the Gym environment is registered."""
    print("\nTesting Gym registration...")
    
    try:
        import gymnasium as gym
        from isaaclab_tasks.manager_based.locomotion.velocity.config import humanoid_abs
        
        env_id = "Isaac-Velocity-Flat-Humanoid-Abs-v0"
        if env_id in gym.envs.registry:
            print(f"✓ Environment '{env_id}' registered successfully")
            return True
        else:
            print(f"✗ Environment '{env_id}' not found in registry")
            return False
    except Exception as e:
        print(f"✗ Failed to check registration: {e}")
        return False

def main():
    print("=" * 60)
    print("Humanoid_abs Configuration Test")
    print("=" * 60)
    
    import_success = test_imports()
    registration_success = test_gym_registration()
    
    print("\n" + "=" * 60)
    if import_success and registration_success:
        print("✓ All tests passed!")
        print("\nYou can now train with:")
        print("  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Humanoid-Abs-v0")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
