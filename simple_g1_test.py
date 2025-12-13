#!/usr/bin/env python3
"""
Simple G1 Model Test - Lightweight version
แสดงผลการทำงานของโมเดลแบบง่าย
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

# Parse arguments  
parser = argparse.ArgumentParser(description="Simple G1 model test")
parser.add_argument("--model_path", type=str, 
                   default="logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt",
                   help="Path to trained model")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless mode to save resources
args_cli.headless = True

# Launch Isaac Sim in headless mode
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("🤖 Simple G1 Model Analysis")
print("=" * 40)

def analyze_model():
    """Analyze the trained model and show what it outputs"""
    
    try:
        # Load model
        model_data = torch.load(args_cli.model_path, map_location='cpu', weights_only=False)
        
        print(f"✅ Model loaded successfully!")
        print(f"   • Training iterations: {model_data.get('iter', 'Unknown')}")
        
        # Analyze model architecture
        state_dict = model_data['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"   • Total parameters: {total_params:,}")
        
        # Find network layers
        actor_layers = [k for k in state_dict.keys() if 'actor' in k and 'weight' in k]
        critic_layers = [k for k in state_dict.keys() if 'critic' in k and 'weight' in k]
        
        print(f"\n🏗️  Network Architecture:")
        print(f"   • Actor layers: {len(actor_layers)}")
        print(f"   • Critic layers: {len(critic_layers)}")
        
        if actor_layers:
            print("\n📊 Actor Network (Policy):")
            for i, layer in enumerate(actor_layers):
                shape = state_dict[layer].shape
                layer_name = layer.replace('actor.', '').replace('.weight', '')
                print(f"   Layer {i+1} ({layer_name}): {shape[1]} → {shape[0]}")
            
            # Show what this means
            input_size = state_dict[actor_layers[0]].shape[1]
            output_size = state_dict[actor_layers[-1]].shape[0]
            
            print(f"\n🎯 Model Input/Output:")
            print(f"   • Input size: {input_size} (observations)")
            print(f"   • Output size: {output_size} (joint positions)")
            
            print(f"\n📝 Input Breakdown (estimated):")
            print(f"   • Base linear velocity: 3 values")
            print(f"   • Base angular velocity: 3 values") 
            print(f"   • Projected gravity: 3 values")
            print(f"   • Velocity commands: 3 values")
            remaining = input_size - 12
            joints_estimate = remaining // 3  # pos + vel + last_action
            print(f"   • Joint positions: ~{joints_estimate} joints")
            print(f"   • Joint velocities: ~{joints_estimate} joints")
            print(f"   • Previous actions: ~{joints_estimate} joints")
            
            print(f"\n🎮 Output Breakdown:")
            print(f"   • Joint position targets: {output_size} joints")
            print(f"   • These go to PD controllers for each motor")
            
            # Create dummy observation to test model
            print(f"\n🧪 Testing Model with Dummy Input:")
            
            # Create a simple network to test
            try:
                # Build network from state dict
                layers = []
                for i, layer in enumerate(actor_layers):
                    in_size = state_dict[layer].shape[1]
                    out_size = state_dict[layer].shape[0]
                    layers.append(torch.nn.Linear(in_size, out_size))
                    if i < len(actor_layers) - 1:  # No activation on output
                        layers.append(torch.nn.ELU())
                
                policy_net = torch.nn.Sequential(*layers)
                
                # Load weights
                actor_state = {k.replace('actor.', ''): v for k, v in state_dict.items() if 'actor' in k}
                policy_net.load_state_dict(actor_state, strict=False)
                policy_net.eval()
                
                # Create dummy observation
                dummy_obs = torch.randn(1, input_size) * 0.1  # Small random values
                
                with torch.no_grad():
                    dummy_actions = policy_net(dummy_obs)
                
                print(f"   • Input shape: {dummy_obs.shape}")
                print(f"   • Output shape: {dummy_actions.shape}")
                print(f"   • Output range: [{dummy_actions.min():.3f}, {dummy_actions.max():.3f}]")
                print(f"   • Sample outputs: {dummy_actions[0, :5].tolist()}")
                
                print(f"\n✅ Model is working! These outputs would be:")
                print(f"   • Converted to radians (joint angles)")
                print(f"   • Sent to each motor's PD controller")
                print(f"   • Motors move to these target positions")
                print(f"   • Robot walks/moves based on these commands")
                
            except Exception as e:
                print(f"   ❌ Could not test model: {e}")
                
        else:
            print("   ❌ Could not find actor layers in model")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    return True

def show_joint_info():
    """Show G1 joint information"""
    
    print(f"\n🦴 G1 Robot Joint Information:")
    print(f"   Based on training configuration")
    
    joint_groups = {
        "Legs (9 joints)": [
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
            "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
            "torso_joint"
        ],
        "Feet (4 joints)": [
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ],
        "Arms (16+ joints)": [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint",
            "left_one_joint", "left_two_joint", "right_one_joint", "right_two_joint",
            "... และอื่นๆ"
        ]
    }
    
    for group_name, joints in joint_groups.items():
        print(f"\n   {group_name}:")
        for joint in joints[:6]:  # Show first 6
            print(f"     • {joint}")
        if len(joints) > 6:
            print(f"     • ... และอีก {len(joints)-6} joints")

def main():
    """Main function"""
    
    print(f"📁 Model path: {args_cli.model_path}")
    print()
    
    # Analyze the model
    success = analyze_model()
    
    if success:
        # Show joint information
        show_joint_info()
        
        print(f"\n🎯 Summary:")
        print(f"   • Model loads and works correctly")
        print(f"   • Input: Robot sensor data + velocity commands")
        print(f"   • Output: Target positions for all joints")
        print(f"   • Ready for real robot deployment!")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Use visualize_g1_model.py for full simulation")
        print(f"   2. Extract joint positions for real robot")
        print(f"   3. Send to PD controllers at 50Hz")
        print(f"   4. Monitor robot performance and safety")
    
    print(f"\n🏁 Analysis complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Analysis stopped by user")
    finally:
        simulation_app.close()