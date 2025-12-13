#!/usr/bin/env python3

"""Test script to verify Humanoid SNC environment registration"""

import gymnasium as gym

def test_env_registration():
    """Test that the Humanoid SNC environment is properly registered"""
    try:
        # Import Isaac Lab tasks to register environments
        import isaaclab_tasks.direct.humanoid_snc
        
        # Check if the environment is registered
        env_id = "Isaac-Humanoid-SNC-Direct-v0"
        
        print(f"✅ Testing environment registration for: {env_id}")
        
        # Try to get environment spec
        spec = gym.spec(env_id)
        print(f"✅ Environment spec found: {spec}")
        
        print(f"✅ Entry point: {spec.entry_point}")
        print(f"✅ Environment successfully registered!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing environment registration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Humanoid SNC Environment Registration")
    print("=" * 50)
    
    success = test_env_registration()
    
    if success:
        print("\n🎉 Environment registration test PASSED!")
        print("\nYou can now use the training command:")
        print("./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-SNC-Direct-v0 --headless --num_envs 1024 --seed 0")
    else:
        print("\n❌ Environment registration test FAILED!")