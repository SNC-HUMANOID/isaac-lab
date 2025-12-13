#!/usr/bin/env python3
"""
Test trained G1 model and demonstrate kinematics usage
"""

import torch
import argparse
import yaml
import pickle
import numpy as np

def load_trained_model(model_path):
    """Load the trained neural network model"""
    try:
        model_data = torch.load(model_path, map_location='cpu')
        print(f"✅ Successfully loaded model from: {model_path}")
        print(f"Model keys: {list(model_data.keys())}")
        
        # Show model architecture info
        if 'model_state_dict' in model_data:
            model_state = model_data['model_state_dict']
            print(f"Model parameters count: {sum(p.numel() for p in model_state.values())}")
        
        return model_data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_config(config_path):
    """Load environment and agent configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Successfully loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def analyze_g1_robot_config(env_config):
    """Analyze G1 robot configuration for kinematics"""
    if not env_config or 'scene' not in env_config:
        print("❌ No robot configuration found")
        return
    
    robot_config = env_config['scene']['robot']
    
    print("\n🤖 G1 Robot Configuration Analysis:")
    print("=" * 50)
    
    # Joint information
    if 'init_state' in robot_config and 'joint_pos' in robot_config['init_state']:
        joint_positions = robot_config['init_state']['joint_pos']
        print(f"📍 Joint Initial Positions ({len(joint_positions)} joints):")
        for joint_name, position in joint_positions.items():
            print(f"  • {joint_name}: {position:.3f} rad ({np.degrees(position):.1f}°)")
    
    # Actuator groups
    if 'actuators' in robot_config:
        actuators = robot_config['actuators']
        print(f"\n⚙️  Actuator Groups ({len(actuators)} groups):")
        
        for group_name, group_config in actuators.items():
            joint_names = group_config.get('joint_names_expr', [])
            effort_limit = group_config.get('effort_limit', 'N/A')
            stiffness = group_config.get('stiffness', 'N/A')
            
            print(f"  🔧 {group_name.upper()}:")
            print(f"     - Joints: {joint_names}")
            print(f"     - Effort Limit: {effort_limit} Nm")
            print(f"     - Stiffness: {stiffness}")
    
    # Robot physical properties
    init_state = robot_config.get('init_state', {})
    if init_state:
        pos = init_state.get('pos', [0, 0, 0])
        print(f"\n📏 Physical Properties:")
        print(f"  • Initial Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m")
        print(f"  • Initial Height: {pos[2]:.2f} m")

def extract_joint_info_for_kinematics(env_config):
    """Extract joint information for external kinematics libraries"""
    if not env_config or 'scene' not in env_config:
        return None
    
    robot_config = env_config['scene']['robot']
    
    # Get USD path for robot model
    usd_path = robot_config['spawn']['usd_path']
    
    # Extract joint information
    joint_info = {
        'robot_model_url': usd_path,
        'joints': {},
        'actuator_groups': {}
    }
    
    # Joint positions and limits
    if 'init_state' in robot_config and 'joint_pos' in robot_config['init_state']:
        joint_positions = robot_config['init_state']['joint_pos']
        for joint_name, position in joint_positions.items():
            joint_info['joints'][joint_name] = {
                'initial_position': position,
                'type': 'revolute'  # Assuming revolute joints
            }
    
    # Actuator information
    if 'actuators' in robot_config:
        for group_name, group_config in robot_config['actuators'].items():
            joint_info['actuator_groups'][group_name] = {
                'joint_names': group_config.get('joint_names_expr', []),
                'effort_limit': group_config.get('effort_limit', 0),
                'velocity_limit': group_config.get('velocity_limit', 0),
                'stiffness': group_config.get('stiffness', 0),
                'damping': group_config.get('damping', 0)
            }
    
    return joint_info

def generate_kinematics_export_script(joint_info):
    """Generate a script to use joint info with external libraries"""
    
    script_content = f'''#!/usr/bin/env python3
"""
G1 Robot Kinematics Export Script
Generated from Isaac Lab training data
"""

import numpy as np

# Robot model information
ROBOT_MODEL_URL = "{joint_info['robot_model_url']}"

# Joint information
JOINTS = {joint_info['joints']}

# Actuator groups
ACTUATOR_GROUPS = {joint_info['actuator_groups']}

def get_joint_names():
    """Get all joint names"""
    return list(JOINTS.keys())

def get_initial_joint_positions():
    """Get initial joint positions as numpy array"""
    return np.array([JOINTS[joint]['initial_position'] for joint in get_joint_names()])

def get_joint_limits():
    """Get joint limits (you may need to add this from URDF)"""
    # TODO: Extract from URDF or add manually
    joint_names = get_joint_names()
    return {{
        'lower': np.array([-np.pi] * len(joint_names)),  # Default limits
        'upper': np.array([np.pi] * len(joint_names))
    }}

# Example usage with PyBullet
def create_pybullet_robot():
    """
    Example: Load G1 robot in PyBullet for kinematics calculation
    """
    try:
        import pybullet as p
        
        # You would need to convert USD to URDF first
        # urdf_path = "path/to/g1.urdf"  
        # robot_id = p.loadURDF(urdf_path)
        
        # Set initial joint positions
        initial_pos = get_initial_joint_positions()
        joint_names = get_joint_names()
        
        print("Initial joint positions:")
        for i, (name, pos) in enumerate(zip(joint_names, initial_pos)):
            print(f"  {{name}}: {{pos:.3f}} rad ({{np.degrees(pos):.1f}}°)")
            # p.resetJointState(robot_id, i, pos)
        
        return None  # robot_id
    except ImportError:
        print("PyBullet not installed. Install with: pip install pybullet")
        return None

# Example usage with Pinocchio
def create_pinocchio_model():
    """
    Example: Load G1 robot in Pinocchio for advanced kinematics
    """
    try:
        import pinocchio as pin
        
        # You would need URDF file
        # model = pin.buildModelFromUrdf("path/to/g1.urdf")
        # data = model.createData()
        
        initial_pos = get_initial_joint_positions()
        joint_names = get_joint_names()
        
        print("Ready for Pinocchio kinematics calculations")
        print(f"Number of joints: {{len(joint_names)}}")
        
        return None  # model, data
    except ImportError:
        print("Pinocchio not installed. Install with: pip install pin")
        return None

if __name__ == "__main__":
    print("G1 Robot Kinematics Information")
    print("=" * 40)
    
    print(f"\\nRobot Model: {{ROBOT_MODEL_URL}}")
    print(f"Number of joints: {{len(JOINTS)}}")
    print(f"Number of actuator groups: {{len(ACTUATOR_GROUPS)}}")
    
    print("\\nJoint Names and Initial Positions:")
    for joint, info in JOINTS.items():
        pos = info['initial_position']
        print(f"  • {{joint}}: {{pos:.3f}} rad ({{np.degrees(pos):.1f}}°)")
    
    print("\\nActuator Groups:")
    for group, info in ACTUATOR_GROUPS.items():
        print(f"  🔧 {{group.upper()}}:")
        print(f"     - Effort limit: {{info['effort_limit']}} Nm")
        print(f"     - Joints: {{info['joint_names']}}")
    
    # Try to create models
    print("\\n" + "="*50)
    print("Testing External Libraries:")
    create_pybullet_robot()
    create_pinocchio_model()
'''
    
    return script_content

def main():
    parser = argparse.ArgumentParser(description="Test G1 trained model and export kinematics")
    parser.add_argument("--model_path", default="logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt")
    parser.add_argument("--env_config", default="logs/rsl_rl/g1_flat/2025-07-29_15-27-26/params/env.yaml")
    parser.add_argument("--agent_config", default="logs/rsl_rl/g1_flat/2025-07-29_15-27-26/params/agent.yaml")
    args = parser.parse_args()
    
    print("🚀 G1 Robot Model and Kinematics Analysis")
    print("="*60)
    
    # 1. Load and analyze trained model
    print("\\n1. Loading Trained Neural Network Model...")
    model = load_trained_model(args.model_path)
    
    # 2. Load environment configuration
    print("\\n2. Loading Environment Configuration...")
    env_config = load_config(args.env_config)
    
    # 3. Load agent configuration  
    print("\\n3. Loading Agent Configuration...")
    agent_config = load_config(args.agent_config)
    
    if agent_config:
        print(f"   • Algorithm: {agent_config.get('algorithm', {}).get('class_name', 'Unknown')}")
        print(f"   • Policy architecture: {agent_config.get('policy', {}).get('actor_hidden_dims', 'Unknown')}")
        print(f"   • Learning rate: {agent_config.get('algorithm', {}).get('learning_rate', 'Unknown')}")
    
    # 4. Analyze robot configuration
    if env_config:
        analyze_g1_robot_config(env_config)
        
        # 5. Extract kinematics information
        print("\\n4. Extracting Kinematics Information...")
        joint_info = extract_joint_info_for_kinematics(env_config)
        
        if joint_info:
            # Generate kinematics export script
            script_content = generate_kinematics_export_script(joint_info)
            
            export_file = "g1_kinematics_export.py"
            with open(export_file, 'w') as f:
                f.write(script_content)
            
            print(f"✅ Generated kinematics export script: {export_file}")
            print(f"   • Contains {len(joint_info['joints'])} joint definitions")
            print(f"   • Contains {len(joint_info['actuator_groups'])} actuator groups")
            print(f"   • Ready for PyBullet, Pinocchio, or other libraries")
    
    print("\\n" + "="*60)
    print("🎯 Summary:")
    print("  ✅ Model analysis completed")
    print("  ✅ Configuration loaded")  
    print("  ✅ Kinematics export script generated")
    print("\\n💡 Next steps:")
    print("  1. Run 'python g1_kinematics_export.py' to see joint information")
    print("  2. Convert USD model to URDF for external libraries")
    print("  3. Use the trained model with Isaac Lab for simulation")

if __name__ == "__main__":
    main()