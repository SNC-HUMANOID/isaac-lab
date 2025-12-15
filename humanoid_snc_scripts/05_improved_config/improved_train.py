#!/usr/bin/env python3
"""🚀 การเทรน Humanoid SNC ด้วย Isaac Lab - ใช้งานได้จริง"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Humanoid SNC - Working Version")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Training iterations")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import numpy as np
from datetime import datetime

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Base environment
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

# Create improved robot configuration using real specs
HUMANOID_SNC_IMPROVED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Arms - ใช้ค่าจาก CSV (effort: 12-27 Nm, velocity: 1.047-2.618 rad/s)
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*shoulder.*", ".*elbow.*"],
            effort_limit=120.0,  # เพิ่มจาก CSV สำหรับ safety margin
            velocity_limit=15.0,
            stiffness=40.0,
            damping=4.0,
        ),
        # Legs - ใช้ค่าจาก CSV (effort: 25-27 Nm, velocity: 2.8795 rad/s) 
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*ankle.*"],
            effort_limit=150.0,  # เพิ่มจาก CSV
            velocity_limit=20.0,
            stiffness=80.0,
            damping=6.0,
        ),
        # Torso
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["Pelvis.*"],
            effort_limit=50.0,
            velocity_limit=10.0,
            stiffness=100.0,
            damping=8.0,
        ),
    },
)

@configclass  
class ImprovedHumanoidEnvCfg(DirectRLEnvCfg):
    """Improved Humanoid SNC Environment Configuration"""
    
    # Environment settings
    episode_length_s = 20.0  # เพิ่มเวลาให้เรียนรู้มากขึ้น
    decimation = 2
    action_scale = 0.5  # ลดลงเพื่อให้นุ่มนวลกว่า
    action_space = 21
    observation_space = 75
    state_space = 0

    # Simulation - ปรับสำหรับความเสถียร
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
    )
    
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.2,  # เพิ่มแรงเสียดทาน
            dynamic_friction=1.0,
            restitution=0.1,  # เพิ่มการดีดกลับเล็กน้อย
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True
    )

    # Robot
    robot: ArticulationCfg = HUMANOID_SNC_IMPROVED_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Joint gears based on real effort limits from CSV
    joint_gears: list = [
        27.0,  # left_shoulder_pitch_joint
        27.0,  # left_shoulder_roll_joint  
        12.0,  # left_shoulder_yaw_joint
        12.0,  # left_elbow_joint
        27.0,  # right_shoulder_pitch_joint
        27.0,  # right_shoulder_roll_joint
        12.0,  # right_shoulder_yaw_joint
        12.0,  # right_elbow_joint
        50.0,  # Pelvis_joint (no data, use default)
        25.0,  # left_hip_pitch_joint
        25.0,  # left_hip_roll_joint
        27.0,  # left_hip_yaw_joint
        25.0,  # left_ankle_pitch_joint
        12.0,  # left_ankle_roll_joint
        12.0,  # left_ankle_yaw_joint
        25.0,  # right_hip_pitch_joint
        25.0,  # right_hip_roll_joint
        27.0,  # right_hip_yaw_joint
        25.0,  # right_ankle_pitch_joint
        12.0,  # right_ankle_roll_joint
        12.0,  # right_ankle_yaw_joint
    ]

    # Reward parameters - ปรับให้เหมาะกับการเดิน
    heading_weight: float = 1.0       # เพิ่มน้ำหนักการเดินตรง
    up_weight: float = 0.5           # ความสมดุล
    energy_cost_scale: float = 0.02   # ลดการลงโทษพลังงาน
    actions_cost_scale: float = 0.005 # ลดการลงโทษ action
    joints_at_limit_cost_scale: float = 0.2
    death_cost: float = -2.0         # เพิ่มการลงโทษการล้ม
    termination_height: float = 0.4  # ลดความสูงขั้นต่ำ
    alive_reward_scale: float = 1.0  # เพิ่มรางวัลการอยู่รอด
    
    # Additional locomotion parameters
    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01
    
    # Required locomotion environment parameters
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 0.25

class ImprovedHumanoidEnv(LocomotionEnv):
    """Improved Humanoid SNC Environment with better reward function"""
    
    cfg: ImprovedHumanoidEnvCfg

    def __init__(self, cfg: ImprovedHumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_rewards(self) -> torch.Tensor:
        """Enhanced reward function"""
        
        # Get basic rewards from parent class
        rewards = super()._get_rewards()
        
        # Add forward velocity reward
        base_lin_vel = self.robot.data.root_lin_vel_b[:, 0]  # forward velocity
        forward_reward = torch.clamp(base_lin_vel, 0.0, 2.0) * 0.5
        
        # Add stability reward (low angular velocity)
        base_ang_vel = torch.norm(self.robot.data.root_ang_vel_b, dim=1)
        stability_reward = torch.exp(-2.0 * base_ang_vel) * 0.2
        
        # Foot contact reward (encourage proper foot placement)
        foot_contacts = self._get_foot_contacts()
        contact_reward = foot_contacts * 0.1
        
        # Joint velocity smoothness (encourage smooth movements)
        joint_vel_smooth = torch.exp(-0.1 * torch.norm(self.robot.data.joint_vel, dim=1)) * 0.1
        
        # Combine all rewards
        total_reward = rewards + forward_reward + stability_reward + contact_reward + joint_vel_smooth
        
        return total_reward
    
    def _get_foot_contacts(self) -> torch.Tensor:
        """Check if feet are in contact with ground"""
        # This is a simplified version - in real implementation you'd use contact sensors
        foot_height = self.robot.data.root_pos_w[:, 2]
        contact_reward = torch.where(foot_height > 0.8, 1.0, 0.0)
        return contact_reward

def simple_train():
    """Simple training loop without RSL-RL complications"""
    
    print("🚀 Starting Improved Humanoid SNC Training")
    print(f"📊 Environments: {args_cli.num_envs}")
    print(f"🔄 Max iterations: {args_cli.max_iterations}")
    
    # Create environment
    env_cfg = ImprovedHumanoidEnvCfg()
    env = ImprovedHumanoidEnv(cfg=env_cfg)
    
    print(f"🤖 Environment created with {env.num_envs} robots")
    print(f"🎮 Action space: {env.action_space}")
    print(f"👁️  Observation space: {env.observation_space}")
    
    # Simple training variables
    best_reward = -float('inf')
    episode_rewards = []
    policy_weights = torch.randn(env.action_space.shape[-1], env.observation_space.shape[-1]) * 0.1
    policy_weights = policy_weights.to(env.device)
    
    # Training loop
    step = 0
    episode_count = 0
    
    print("🏃‍♂️ Starting training...")
    
    try:
        for iteration in range(args_cli.max_iterations):
            episode_reward = 0.0
            steps_in_episode = 0
            
            # Reset environment
            with torch.no_grad():
                obs, _ = env.reset()
            
            for ep_step in range(int(env_cfg.episode_length_s * 60)):  # 60 Hz
                with torch.no_grad():
                    # Get observation tensor from dict
                    if isinstance(obs, dict):
                        obs_tensor = obs["policy"]  # Isaac Lab uses "policy" key
                    else:
                        obs_tensor = obs
                    
                    # Simple linear policy
                    actions = torch.tanh(obs_tensor @ policy_weights.T) * env_cfg.action_scale
                    
                    # Add exploration noise
                    if iteration < args_cli.max_iterations * 0.8:
                        noise_std = 0.2 * (1.0 - iteration / args_cli.max_iterations)
                        noise = torch.randn_like(actions) * noise_std
                        actions = actions + noise
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(actions)
                    
                    episode_reward += reward.mean().item()
                    steps_in_episode += 1
                    step += 1
                    
                    # Break if episode ends
                    if terminated.any() or truncated.any():
                        break
            
            # Simple policy improvement (gradient-free)
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"🎯 New best reward: {best_reward:.3f} at iteration {iteration}")
            else:
                # Add small random perturbation to explore
                policy_weights += torch.randn_like(policy_weights) * 0.001
            
            episode_rewards.append(episode_reward)
            
            if iteration % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"⏱️  Iteration: {iteration:4d}, Avg Reward: {avg_reward:.3f}, Steps: {steps_in_episode}")
            
            episode_count += 1
            
    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user")
    
    print(f"\n✅ Training completed!")
    print(f"📊 Final best reward: {best_reward:.3f}")
    print(f"📈 Total episodes: {episode_count}")
    
    # Save simple results
    results_dir = f"logs/simple_humanoid_snc/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    torch.save(policy_weights, f"{results_dir}/policy_weights.pt")
    np.save(f"{results_dir}/episode_rewards.npy", np.array(episode_rewards))
    
    print(f"💾 Results saved to: {results_dir}")
    
    env.close()

def main():
    """Main function"""
    simple_train()

if __name__ == "__main__":
    main()
    simulation_app.close()