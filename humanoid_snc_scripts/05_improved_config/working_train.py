#!/usr/bin/env python3
"""🚀 Working Humanoid SNC training script - proper Isaac Lab structure"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Working Humanoid SNC Training")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=100, help="Training iterations")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from datetime import datetime

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Robot configuration
HUMANOID_SNC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),  # เพิ่มความสูงให้มากขึ้น
        rot=(1.0, 0.0, 0.0, 0.0),  # แน่ใจว่าไม่เอียง
        # ปรับ joint positions เพื่อให้ยืนได้ดีขึ้น (ใช้เฉพาะค่า default)
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # ใช้ actuator เดียวเพื่อหลีกเลี่ยงปัญหา pattern ซ้ำซ้อน
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=150.0,    # แรงพอสำหรับการยืน
            velocity_limit=10.0,
            stiffness=80.0,        # ความแข็งสำหรับการยืน
            damping=8.0,           # damping เพื่อความเสถียร
        ),
    },
)

@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Humanoid Environment Configuration"""
    
    # Environment settings
    episode_length_s = 10.0
    decimation = 2
    action_scale = 0.25
    action_space = 21
    observation_space = 30  # Robot state observation
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
    )

    # Robot
    robot_cfg: ArticulationCfg = HUMANOID_SNC_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True
    )

    # Reward scales
    height_reward_scale: float = 1.0
    joint_penalty_scale: float = 0.01
    stability_reward_scale: float = 0.5

class HumanoidEnv(DirectRLEnv):
    """Humanoid Environment"""
    
    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Set up the scene with robot and ground"""
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step with standing assistance"""
        
        # เป้าหมายการยืน (standing target pose)
        target_joint_pos = torch.zeros_like(self.robot.data.joint_pos)
        
        # ถ้าหาชื่อ joint ได้ ให้ตั้งค่าเป้าหมาย
        try:
            hip_pitch_indices = []
            ankle_pitch_indices = []
            
            # หา indices ของ joint ที่สำคัญ
            for joint_name in self.robot.joint_names:
                if "hip" in joint_name.lower() and "pitch" in joint_name.lower():
                    idx = self.robot.joint_names.index(joint_name)
                    hip_pitch_indices.append(idx)
                elif "ankle" in joint_name.lower() and "pitch" in joint_name.lower():
                    idx = self.robot.joint_names.index(joint_name)
                    ankle_pitch_indices.append(idx)
            
            # ตั้งเป้าหมายสำหรับการยืน
            for idx in hip_pitch_indices:
                target_joint_pos[:, idx] = -0.1  # เอียงสะโพกไปข้างหลัง
            for idx in ankle_pitch_indices:
                target_joint_pos[:, idx] = 0.1   # เอียงข้อเท้าไปข้างหน้า
                
        except:
            # ถ้าหา joint ไม่ได้ ให้ใช้ค่า 0
            pass
        
        # PD control สำหรับช่วยยืน
        kp_standing = 20.0  # Proportional gain สำหรับการยืน
        kd_standing = 2.0   # Derivative gain สำหรับการยืน
        
        pos_error = target_joint_pos - self.robot.data.joint_pos
        vel_error = -self.robot.data.joint_vel  # ต้องการ velocity = 0
        
        standing_torque = kp_standing * pos_error + kd_standing * vel_error
        
        # รวม action จากผู้ใช้กับการช่วยยืน
        user_actions = self.cfg.action_scale * actions.clone()
        self.actions = standing_torque + user_actions
        
        # จำกัดแรง
        self.actions = torch.clamp(self.actions, -200.0, 200.0)

    def _apply_action(self) -> None:
        """Apply actions to robot"""
        self.robot.set_joint_effort_target(self.actions)

    def _get_observations(self) -> dict:
        """Get observations from environment"""
        # Basic observation: joint positions, joint velocities, root orientation
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel * 0.1  # Scale velocities
        root_quat = self.robot.data.root_quat_w  # World frame quaternion
        
        # Combine observations (21 joint pos + 21 joint vel, but limit to 30 total)
        obs = torch.cat([
            joint_pos[:, :15],  # First 15 joint positions
            joint_vel[:, :15],  # First 15 joint velocities
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Calculate reward function focused on standing"""
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        
        # รางวัลความสูง (สำคัญมาก - ต้องอยู่สูงจากพื้น)
        height_target = 1.2  # เป้าหมาย 1.2m
        height_reward = torch.clamp(root_pos[:, 2] - 0.5, 0.0, 1.0) * 3.0  # รางวัลสูง
        
        # รางวัลการยืนตรง (ใช้ quaternion w component)
        upright_reward = (root_quat[:, 0] ** 2) * 2.0  # w component = 1 เมื่อยืนตรง
        
        # ลงโทษการเอียง (x, y components ของ quaternion)
        tilt_penalty = -torch.norm(root_quat[:, 1:3], dim=1) * 1.0
        
        # รางวัลความเสถียร (angular velocity ต่ำ)
        root_ang_vel = self.robot.data.root_ang_vel_w
        stability_reward = torch.exp(-torch.norm(root_ang_vel, dim=1)) * 1.0
        
        # ลงโทษการเคลื่อนไหวมากเกินไป
        joint_penalty = -0.01 * torch.sum(torch.abs(self.robot.data.joint_pos), dim=1)
        velocity_penalty = -0.01 * torch.sum(torch.abs(self.robot.data.joint_vel), dim=1)
        
        # รวมรางวัล (เน้นการยืน)
        total_reward = height_reward + upright_reward + tilt_penalty + stability_reward + joint_penalty + velocity_penalty
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Episode termination conditions"""
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        
        # ยุติถ้าหุ่นยนต์ล้ม (ต่ำเกินไป)
        height_termination = root_pos[:, 2] < 0.4  # ยุติถ้าต่ำกว่า 40cm
        
        # ยุติถ้าเอียงมากเกินไป (ใช้ quaternion w component)
        tilt_termination = root_quat[:, 0] < 0.7  # cos(45°) ≈ 0.7
        
        # รวมเงื่อนไข termination
        terminated = height_termination | tilt_termination
        
        # Time-based truncation
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments to standing position"""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        # Reset robot state ในท่ายืน
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 2] = 1.5  # ตั้งความสูง 1.5m
        default_root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # quaternion สำหรับยืนตรง
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        
        # Reset joint states ในท่าที่เหมาะกับการยืน
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        
        # ปรับ joint positions เพื่อการยืน
        try:
            for i, joint_name in enumerate(self.robot.joint_names):
                if "hip" in joint_name.lower() and "pitch" in joint_name.lower():
                    default_joint_pos[:, i] = -0.1  # เอียงสะโพกไปข้างหลัง
                elif "ankle" in joint_name.lower() and "pitch" in joint_name.lower():
                    default_joint_pos[:, i] = 0.1   # เอียงข้อเท้าไปข้างหน้า
        except:
            pass
        
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)

def simple_train():
    """Simple training function"""
    
    print("🚀 Working Humanoid SNC Training")
    print(f"📊 Environments: {args_cli.num_envs}")
    print(f"🔄 Max iterations: {args_cli.max_iterations}")
    
    # Create environment
    env_cfg = HumanoidEnvCfg()
    env = HumanoidEnv(cfg=env_cfg)
    
    print(f"🤖 Environment created with {env.num_envs} robots")
    print(f"🎮 Action space: {env.action_space}")
    print(f"👁️  Observation space: {env.observation_space}")
    
    # Simple training parameters
    best_reward = -float('inf')
    episode_rewards = []
    
    # Random policy that will improve over time
    policy_mean = torch.zeros(21, device=env.device)
    policy_std = torch.ones(21, device=env.device) * 0.2
    
    print("🏃‍♂️ Starting training...")
    
    for iteration in range(args_cli.max_iterations):
        episode_reward = 0.0
        steps_in_episode = 0
        
        # Reset environment
        obs, _ = env.reset()
        
        # Episode loop
        for step in range(int(env_cfg.episode_length_s * 30)):  # 30 Hz
            # Generate actions from simple policy (เล็กๆ เพื่อช่วยปรับสมดุล)
            actions = torch.normal(policy_mean, policy_std).unsqueeze(0).repeat(env.num_envs, 1)
            actions = torch.clamp(actions, -0.3, 0.3)  # จำกัดให้เล็กลง
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            
            episode_reward += reward.mean().item()
            steps_in_episode += 1
            
            if terminated.any() or truncated.any():
                break
        
        # Simple policy improvement - reduce noise over time
        avg_reward = episode_reward / max(steps_in_episode, 1)
        episode_rewards.append(avg_reward)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            # Slight improvement: reduce exploration noise
            policy_std *= 0.99
            
        # Progress decay of exploration
        policy_std = torch.clamp(policy_std, 0.05, 0.3)
        
        if iteration % 5 == 0:  # รายงานบ่อยขึ้น
            recent_avg = np.mean(episode_rewards[-5:])
            final_height = env.robot.data.root_pos_w[:, 2].mean().item()
            print(f"⏱️  Iteration: {iteration:4d}, Reward: {avg_reward:.3f}, Height: {final_height:.3f}m, Best: {best_reward:.3f}")
            
            if final_height > 0.8:
                print(f"    ✅ หุ่นยนต์ยืนได้! ความสูง: {final_height:.3f}m")
            elif final_height > 0.5:
                print(f"    📈 หุ่นยนต์กำลังลุกขึ้น: {final_height:.3f}m")
            else:
                print(f"    ❌ หุ่นยนต์ยังนอนอยู่: {final_height:.3f}m")
    
    print(f"✅ Training completed!")
    print(f"📊 Best reward: {best_reward:.3f}")
    print(f"📈 Final average: {np.mean(episode_rewards[-10:]):.3f}")
    
    # Save simple results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = f"logs/working_humanoid_snc_{timestamp}.npy"
    np.save(results_path, np.array(episode_rewards))
    print(f"💾 Results saved to: {results_path}")
    
    env.close()

def main():
    """Main function"""
    simple_train()

if __name__ == "__main__":
    main()
    simulation_app.close()