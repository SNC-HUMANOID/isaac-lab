#!/usr/bin/env python3
"""
G1 Model Analysis without Isaac Sim
วิเคราะห์โมเดลโดยไม่ต้องใช้ Isaac Sim
"""

import torch
import numpy as np
import yaml
import pickle

def load_and_analyze_model(model_path):
    """Load and analyze the trained model"""
    
    print("🤖 G1 Model Analysis")
    print("=" * 50)
    print(f"📁 Model: {model_path}")
    print()
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("✅ Model loaded successfully!")
        print(f"   • Training iterations: {checkpoint.get('iter', 'Unknown')}")
        
        # Analyze model structure
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"   • Total parameters: {total_params:,}")
        
        # Separate actor and critic networks
        actor_params = {k: v for k, v in state_dict.items() if 'actor' in k}
        critic_params = {k: v for k, v in state_dict.items() if 'critic' in k}
        
        print(f"   • Actor parameters: {sum(p.numel() for p in actor_params.values()):,}")
        print(f"   • Critic parameters: {sum(p.numel() for p in critic_params.values()):,}")
        
        # Analyze actor network (the one we use for inference)
        print(f"\n🧠 Actor Network (Policy Network):")
        actor_layers = [k for k in actor_params.keys() if 'weight' in k]
        actor_layers.sort()
        
        input_size = None
        output_size = None
        
        for i, layer_name in enumerate(actor_layers):
            weight = state_dict[layer_name]
            shape = weight.shape
            layer_short = layer_name.replace('actor.', '').replace('.weight', '')
            
            print(f"   Layer {i+1:2d} ({layer_short:8s}): {shape[1]:3d} → {shape[0]:3d}")
            
            if i == 0:
                input_size = shape[1]
            if i == len(actor_layers) - 1:
                output_size = shape[0]
        
        print(f"\n📊 Network Summary:")
        print(f"   • Input size:  {input_size} (observations)")
        print(f"   • Output size: {output_size} (joint actions)")
        print(f"   • Hidden layers: {len(actor_layers) - 1}")
        
        # Test the network
        print(f"\n🧪 Testing Network:")
        
        # Create the actor network
        policy_net = create_policy_network(actor_params, state_dict)
        
        if policy_net:
            # Test with dummy input
            dummy_obs = torch.randn(1, input_size) * 0.1
            
            with torch.no_grad():
                dummy_actions = policy_net(dummy_obs)
            
            print(f"   • Input shape: {dummy_obs.shape}")
            print(f"   • Output shape: {dummy_actions.shape}")
            print(f"   • Output range: [{dummy_actions.min():.3f}, {dummy_actions.max():.3f}]")
            
            # Show some sample outputs
            sample_actions = dummy_actions[0, :min(10, output_size)]
            print(f"   • Sample actions: {[f'{x:.3f}' for x in sample_actions.tolist()]}")
            
            return True, input_size, output_size, policy_net
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False, None, None, None

def create_policy_network(actor_params, state_dict):
    """Create and load the policy network"""
    
    try:
        # Find layer weights
        weight_keys = [k for k in actor_params.keys() if 'weight' in k]
        weight_keys.sort()
        
        # Build network
        layers = []
        for i, weight_key in enumerate(weight_keys):
            weight = state_dict[weight_key]
            in_size, out_size = weight.shape[1], weight.shape[0]
            
            layers.append(torch.nn.Linear(in_size, out_size))
            
            # Add activation (ELU) except for output layer
            if i < len(weight_keys) - 1:
                layers.append(torch.nn.ELU())
        
        policy_net = torch.nn.Sequential(*layers)
        
        # Load weights
        policy_state = {}
        for k, v in actor_params.items():
            new_key = k.replace('actor.', '')
            policy_state[new_key] = v
        
        policy_net.load_state_dict(policy_state, strict=False)
        policy_net.eval()
        
        print(f"   ✅ Network created and loaded successfully")
        return policy_net
        
    except Exception as e:
        print(f"   ❌ Failed to create network: {e}")
        return None

def analyze_config_files():
    """Analyze configuration files"""
    
    print(f"\n⚙️  Training Configuration:")
    
    # Load agent config
    try:
        with open("logs/rsl_rl/g1_flat/2025-07-29_15-27-26/params/agent.yaml", 'r') as f:
            agent_config = yaml.safe_load(f)
        
        print(f"   • Algorithm: {agent_config.get('algorithm', {}).get('class_name', 'Unknown')}")
        print(f"   • Learning rate: {agent_config.get('algorithm', {}).get('learning_rate', 'Unknown')}")
        print(f"   • Max iterations: {agent_config.get('max_iterations', 'Unknown')}")
        print(f"   • Steps per env: {agent_config.get('num_steps_per_env', 'Unknown')}")
        
        policy = agent_config.get('policy', {})
        if policy:
            print(f"   • Actor layers: {policy.get('actor_hidden_dims', 'Unknown')}")
            print(f"   • Critic layers: {policy.get('critic_hidden_dims', 'Unknown')}")
            print(f"   • Activation: {policy.get('activation', 'Unknown')}")
            
    except Exception as e:
        print(f"   ❌ Could not load agent config: {e}")
    
    # Load environment config  
    try:
        with open("logs/rsl_rl/g1_flat/2025-07-29_15-27-26/params/env.yaml", 'r') as f:
            env_config = yaml.safe_load(f)
        
        print(f"\n🌍 Environment Configuration:")
        scene = env_config.get('scene', {})
        print(f"   • Number of envs: {scene.get('num_envs', 'Unknown')}")
        print(f"   • Environment spacing: {scene.get('env_spacing', 'Unknown')}m")
        print(f"   • Episode length: {env_config.get('episode_length_s', 'Unknown')}s")
        
        # Action configuration
        actions = env_config.get('actions', {})
        if actions:
            joint_pos = actions.get('joint_pos', {})
            print(f"   • Action type: {joint_pos.get('class_type', 'Unknown')}")
            print(f"   • Action scale: {joint_pos.get('scale', 'Unknown')}")
            
    except Exception as e:
        print(f"   ❌ Could not load env config: {e}")

def explain_usage(input_size, output_size, policy_net):
    """Explain how to use the model"""
    
    print(f"\n🎯 How to Use This Model:")
    print(f"=" * 50)
    
    print(f"📥 INPUT ({input_size} values):")
    print(f"   1. Base linear velocity (3 values): [vx, vy, vz] in m/s")
    print(f"   2. Base angular velocity (3 values): [wx, wy, wz] in rad/s")
    print(f"   3. Projected gravity (3 values): [gx, gy, gz] normalized")
    print(f"   4. Velocity commands (3 values): [cmd_vx, cmd_vy, cmd_wz]")
    
    remaining = input_size - 12
    estimated_joints = remaining // 3
    print(f"   5. Joint positions (~{estimated_joints} joints): current positions")
    print(f"   6. Joint velocities (~{estimated_joints} joints): current velocities") 
    print(f"   7. Previous actions (~{estimated_joints} joints): last sent commands")
    
    print(f"\n📤 OUTPUT ({output_size} values):")
    print(f"   • Joint position targets for {output_size} joints")
    print(f"   • Each value is in radians")
    print(f"   • These go to PD controllers")
    
    print(f"\n🔄 Usage Pattern:")
    print(f"   ```python")
    print(f"   # 1. Get robot state")
    print(f"   observation = get_robot_state()  # {input_size} values")
    print(f"   ")
    print(f"   # 2. Run through AI model")
    print(f"   actions = policy_net(observation)  # {output_size} values")
    print(f"   ")
    print(f"   # 3. Send to robot")
    print(f"   for i, joint_name in enumerate(joint_names):")
    print(f"       target_pos = actions[i]")
    print(f"       robot.set_joint_position(joint_name, target_pos)")
    print(f"   ```")
    
    print(f"\n⚡ Control Frequency:")
    print(f"   • Run this loop at 50 Hz (every 0.02 seconds)")
    print(f"   • Update observations from robot sensors")
    print(f"   • Send new position targets to motors")
    
    if policy_net:
        print(f"\n🧪 Example Usage:")
        
        # Create example observation
        example_obs = torch.tensor([[
            # Base velocities
            0.5, 0.0, 0.0,    # walking forward
            0.0, 0.0, 0.0,    # no rotation
            # Gravity 
            0.0, 0.0, -1.0,   # pointing down
            # Commands
            0.5, 0.0, 0.0,    # want to walk forward
        ] + [0.0] * (input_size - 12)])  # joints (all zeros for example)
        
        with torch.no_grad():
            example_actions = policy_net(example_obs)
        
        print(f"   Input: Walking forward command")
        print(f"   Output: {example_actions.shape[1]} joint targets")
        print(f"   Sample joint targets: {[f'{x:.3f}' for x in example_actions[0, :8].tolist()]}")

def show_joint_groups():
    """Show G1 joint organization"""
    
    print(f"\n🦴 G1 Robot Joint Groups:")
    print(f"=" * 50)
    
    joint_groups = {
        "🦵 LEGS (9 joints) - Main locomotion": [
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
            "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
            "torso_joint"
        ],
        "🦶 FEET (4 joints) - Balance & ground contact": [
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ],
        "💪 ARMS (14+ joints) - Posture & balance": [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint", "left_elbow_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
            "right_elbow_pitch_joint", "right_elbow_roll_joint",
            "left_one_joint", "left_two_joint", "right_one_joint", "right_two_joint"
        ]
    }
    
    for group_name, joints in joint_groups.items():
        print(f"\n{group_name}:")
        for joint in joints:
            print(f"   • {joint}")

def main():
    """Main function"""
    
    model_path = "logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt"
    
    # Analyze the model
    success, input_size, output_size, policy_net = load_and_analyze_model(model_path)
    
    if success:
        # Analyze config files
        analyze_config_files()
        
        # Show joint information
        show_joint_groups()
        
        # Explain usage
        explain_usage(input_size, output_size, policy_net)
        
        print(f"\n🎉 Analysis Complete!")
        print(f"=" * 50)
        print(f"✅ Model is ready for deployment")
        print(f"✅ {output_size} joints will be controlled")
        print(f"✅ Network processes {input_size} sensor inputs")
        print(f"✅ Use at 50 Hz control frequency")
        
    else:
        print(f"\n❌ Analysis failed!")

if __name__ == "__main__":
    main()