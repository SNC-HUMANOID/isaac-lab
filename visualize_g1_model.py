#!/usr/bin/env python3
"""
G1 Model Visualization Script
แสดงการทำงานของโมเดล AI ที่เทรนแล้วใน Isaac Sim
"""

import argparse
import torch
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize G1 trained model")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--model_path", type=str, 
                   default="logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt",
                   help="Path to trained model")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after Isaac Sim launch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
import isaaclab.utils.math as math_utils

@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Configuration for G1 visualization scene."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # G1 Robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
                ".*_hip_pitch_joint": -0.2,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                ".*_elbow_pitch_joint": 0.87,
                "left_shoulder_roll_joint": 0.16,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_pitch_joint": 0.35,
                "left_one_joint": 1.0,
                "right_one_joint": -1.0,
                "left_two_joint": 0.52,
                "right_two_joint": -0.52,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", "torso_joint"],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                    "torso_joint": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                    "torso_joint": 5.0,
                },
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_.*_joint"],
                effort_limit=20,
                stiffness=20.0,
                damping=2.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_.*_joint", ".*_.*_joint"],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness=40.0,
                damping=10.0,
            ),
        },
    )
    
    # Contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.005,
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )
    
    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

class G1ModelVisualization:
    """G1 Model Visualization Class."""
    
    def __init__(self, model_path: str, num_envs: int = 1):
        """Initialize visualization."""
        
        # Store parameters
        self.num_envs = num_envs
        self.model_path = model_path
        
        # Initialize simulation context first
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0" if torch.cuda.is_available() else "cpu")
        self.sim = SimulationContext(sim_cfg)
        
        # Create scene
        scene_cfg = G1SceneCfg(num_envs=num_envs, env_spacing=2.5)
        self.scene = InteractiveScene(scene_cfg)
        
        # Get robot asset
        self.robot = self.scene["robot"]
        self.contact_sensor = self.scene["contact_forces"]
        
        # Load trained model
        self.load_model()
        
        # Setup joint information
        self.setup_joints()
        
        # Control parameters
        self.action_scale = 0.5
        self.control_freq = 50  # Hz
        self.dt = 1.0 / self.control_freq
        
        # Command variables
        self.velocity_command = torch.zeros(3, device=self.robot.device)  # [vx, vy, wz]
        self.step_count = 0
        
        print(f"🤖 G1 Model Visualization Ready!")
        print(f"   • Model: {model_path}")
        print(f"   • Environments: {num_envs}")
        print(f"   • Joints: {self.robot.num_joints}")
        print(f"   • Action dim: {len(self.action_joint_ids)}")
    
    def load_model(self):
        """Load the trained RL model."""
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.robot.device, weights_only=False)
            
            # Extract model info
            self.model_state_dict = checkpoint["model_state_dict"]
            self.training_iter = checkpoint.get("iter", "Unknown")
            
            # Create a simple neural network (Actor network)
            self.setup_policy_network()
            
            print(f"✅ Model loaded successfully!")
            print(f"   • Training iterations: {self.training_iter}")
            print(f"   • Model parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            # Create dummy network for testing
            self.setup_dummy_network()
    
    def setup_policy_network(self):
        """Setup policy network from checkpoint."""
        
        # Find actor network layers
        actor_layers = [k for k in self.model_state_dict.keys() if 'actor' in k and 'weight' in k]
        
        if actor_layers:
            # Get network architecture from state dict
            input_size = self.model_state_dict[actor_layers[0]].shape[1]
            hidden_sizes = []
            for layer in actor_layers[:-1]:
                hidden_sizes.append(self.model_state_dict[layer].shape[0])
            output_size = self.model_state_dict[actor_layers[-1]].shape[0]
            
            # Create network
            layers = []
            sizes = [input_size] + hidden_sizes + [output_size]
            
            for i in range(len(sizes) - 1):
                layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
                if i < len(sizes) - 2:  # No activation on output layer
                    layers.append(torch.nn.ELU())
            
            self.policy_net = torch.nn.Sequential(*layers).to(self.robot.device)
            
            # Load weights
            self.policy_net.load_state_dict({
                k.replace('actor.', ''): v for k, v in self.model_state_dict.items() 
                if 'actor' in k
            }, strict=False)
            
            self.policy_net.eval()
            
            print(f"   • Network: {input_size} → {hidden_sizes} → {output_size}")
        else:
            self.setup_dummy_network()
    
    def setup_dummy_network(self):
        """Setup dummy network for testing."""
        # Estimate observation and action sizes
        obs_size = 3 + 3 + 3 + 3 + self.robot.num_joints * 2 + self.robot.num_joints  # sensors + joints + actions
        action_size = self.robot.num_joints
        
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, action_size),
        ).to(self.robot.device)
        
        print(f"   • Using dummy network: {obs_size} → [256,128,128] → {action_size}")
    
    def setup_joints(self):
        """Setup joint information."""
        
        # Get all actuated joint IDs and names
        self.action_joint_ids = []
        self.action_joint_names = []
        
        # Add leg joints
        for joint_name in self.robot.joint_names:
            if any(pattern in joint_name for pattern in [
                "_hip_", "_knee_", "torso_",
                "_ankle_", "_shoulder_", "_elbow_", 
                "_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_"
            ]):
                joint_id = self.robot.joint_names.index(joint_name)
                self.action_joint_ids.append(joint_id)
                self.action_joint_names.append(joint_name)
        
        self.action_joint_ids = torch.tensor(self.action_joint_ids, device=self.robot.device)
        
        # Store default positions
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.last_actions = torch.zeros(self.num_envs, len(self.action_joint_ids), device=self.robot.device)
        
        print(f"   • Action joints: {len(self.action_joint_ids)}")
        for i, name in enumerate(self.action_joint_names[:10]):  # Show first 10
            print(f"     - {i}: {name}")
        if len(self.action_joint_names) > 10:
            print(f"     - ... และอีก {len(self.action_joint_names) - 10} joints")
    
    def get_observations(self):
        """Get current observations for the policy."""
        
        # Base velocity (in robot frame)
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        
        # Projected gravity
        projected_gravity = math_utils.quat_rotate_inverse(
            self.robot.data.root_quat_w, 
            torch.tensor([0.0, 0.0, -1.0], device=self.robot.device).repeat(self.num_envs, 1)
        )
        
        # Joint states (relative to default)
        joint_pos = self.robot.data.joint_pos[:, self.action_joint_ids] - self.default_joint_pos[:, self.action_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.action_joint_ids]
        
        # Velocity commands (repeated for current command)
        velocity_commands = self.velocity_command.unsqueeze(0).repeat(self.num_envs, 1)
        
        # Combine observations
        obs = torch.cat([
            base_lin_vel,           # 3
            base_ang_vel,           # 3  
            projected_gravity,      # 3
            velocity_commands,      # 3
            joint_pos,              # num_action_joints
            joint_vel,              # num_action_joints
            self.last_actions,      # num_action_joints
        ], dim=1)
        
        return obs
    
    def compute_actions(self, obs):
        """Compute actions using the policy network."""
        with torch.no_grad():
            raw_actions = self.policy_net(obs)
            
            # Apply scaling and offset
            processed_actions = self.default_joint_pos[:, self.action_joint_ids] + raw_actions * self.action_scale
            
            return processed_actions
    
    def apply_actions(self, actions):
        """Apply actions to the robot."""
        
        # Set joint position targets
        joint_pos_targets = self.robot.data.joint_pos.clone()
        joint_pos_targets[:, self.action_joint_ids] = actions
        
        self.robot.set_joint_position_target(joint_pos_targets)
        
        # Store last actions for next observation
        self.last_actions = actions - self.default_joint_pos[:, self.action_joint_ids]
    
    def update_commands(self):
        """Update velocity commands (simple pattern)."""
        
        time = self.step_count * self.dt
        
        # Create simple walking pattern
        if time < 5.0:
            # Forward walking
            self.velocity_command = torch.tensor([0.5, 0.0, 0.0], device=self.robot.device)
        elif time < 10.0:
            # Turn left
            self.velocity_command = torch.tensor([0.3, 0.0, 0.5], device=self.robot.device) 
        elif time < 15.0:
            # Backward walking
            self.velocity_command = torch.tensor([-0.3, 0.0, 0.0], device=self.robot.device)
        elif time < 20.0:
            # Side step right
            self.velocity_command = torch.tensor([0.0, -0.3, 0.0], device=self.robot.device)
        else:
            # Stop
            self.velocity_command = torch.tensor([0.0, 0.0, 0.0], device=self.robot.device)
    
    def run_visualization(self):
        """Run the visualization loop."""
        
        # Define simulation stepping
        sim_dt = sim_utils.SimulationCfg.dt
        control_decimation = int(self.dt / sim_dt)
        
        # Simulation loop
        count = 0
        while simulation_app.is_running():
            
            # Update commands periodically
            if count % (control_decimation * 10) == 0:  # Every 10 control steps
                self.update_commands()
                print(f"Step {self.step_count:4d} | Time: {self.step_count * self.dt:6.2f}s | Cmd: [{self.velocity_command[0]:.2f}, {self.velocity_command[1]:.2f}, {self.velocity_command[2]:.2f}]")
            
            # Control loop
            if count % control_decimation == 0:
                
                # Get observations
                obs = self.get_observations()
                
                # Compute actions using policy
                actions = self.compute_actions(obs)
                
                # Apply actions
                self.apply_actions(actions)
                
                self.step_count += 1
            
            # Step the simulation
            self.scene.update(sim_dt)
            count += 1

def main():
    """Main function."""
    
    # Create visualization
    viz = G1ModelVisualization(
        model_path=args_cli.model_path,
        num_envs=args_cli.num_envs
    )
    
    # Run visualization
    try:
        viz.run_visualization()
    except KeyboardInterrupt:
        print("\n🛑 Visualization stopped by user")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()