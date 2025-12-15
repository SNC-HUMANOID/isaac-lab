#!/usr/bin/env python3
"""🚀 แก้ปัญหาหุ่นยนต์นอนใน Isaac Sim - ให้ยืนได้"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fix Standing Humanoid Robot")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# แก้ไขการตั้งค่าหุ่นยนต์ให้ยืนได้
HUMANOID_SNC_STANDING_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,  # เพิ่มความเสถียร
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,  # เพิ่มความแม่นยำ
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.2),  # เพิ่มความสูง
        rot=(1.0, 0.0, 0.0, 0.0),  # แน่ใจว่าหุ่นยนต์ยืนตรง
        # ปรับค่า joint เริ่มต้นให้เหมาะกับการยืน
        joint_pos={
            ".*hip_pitch.*": 0.1,      # เอียงสะโพกเล็กน้อย
            ".*hip_roll.*": 0.0,       # ไม่เอียงข้าง
            ".*hip_yaw.*": 0.0,        # ไม่หมุน
            ".*ankle_pitch.*": -0.1,   # เอียงข้อเท้าเพื่อสมดุล
            ".*ankle_roll.*": 0.0,
            ".*ankle_yaw.*": 0.0,
            ".*shoulder.*": 0.0,       # แขนอยู่ข้าง
            ".*elbow.*": 0.0,          # ข้อศอกตรง
            ".*": 0.0,                 # อื่นๆ เป็น 0
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # ปรับ actuator ให้แข็งแรงพอที่จะรองรับน้ำหนักตัว
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*ankle.*"],
            effort_limit=200.0,  # เพิ่มแรง
            velocity_limit=10.0,
            stiffness=100.0,     # เพิ่มความแข็ง
            damping=8.0,         # เพิ่ม damping
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*shoulder.*", ".*elbow.*"],
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness=60.0,
            damping=6.0,
        ),
    },
)

@configclass
class StandingRobotCfg(DirectRLEnvCfg):
    """การตั้งค่าสำหรับหุ่นยนต์ที่ยืนได้"""
    
    # Environment settings
    episode_length_s = 30.0  # ทำให้ยาวขึ้นเพื่อดูผล
    decimation = 2
    action_scale = 0.1  # ลดให้น้อยกว่าเดิม เพื่อการเคลื่อนไหวนุ่มนวล
    action_space = 21
    observation_space = 30
    state_space = 0

    # Simulation - ลด timestep เพื่อความเสถียร
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,  # เพิ่มความละเอียด
        render_interval=decimation,
        substeps=2,  # เพิ่ม substeps
    )

    # Robot
    robot_cfg: ArticulationCfg = HUMANOID_SNC_STANDING_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=6.0,  # เพิ่มระยะห่าง
        replicate_physics=True
    )

class StandingRobotEnv(DirectRLEnv):
    """Environment ที่ช่วยให้หุ่นยนต์ยืนได้"""
    
    cfg: StandingRobotCfg

    def __init__(self, cfg: StandingRobotCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Setup scene"""
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Ground plane with better friction
        ground_cfg = GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average", 
                static_friction=1.5,  # เพิ่มแรงเสียดทาน
                dynamic_friction=1.2,
                restitution=0.0,     # ไม่เด้ง
            )
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
        
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        self.scene.articulations["robot"] = self.robot
        
        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply PD control to maintain standing pose"""
        # รวม action จากผู้ใช้กับการควบคุม PD เพื่อยืน
        
        # Target standing pose (เป้าหมายการยืน)
        standing_pose = torch.zeros_like(self.robot.data.joint_pos)
        standing_pose[:, self.robot.find_joints(".*hip_pitch.*")[0]] = 0.1   # เอียงสะโพกเล็กน้อย
        standing_pose[:, self.robot.find_joints(".*ankle_pitch.*")[0]] = -0.1 # เอียงข้อเท้า
        
        # PD control gains
        kp = 50.0  # Proportional gain
        kd = 5.0   # Derivative gain
        
        # คำนวณ PD control
        pos_error = standing_pose - self.robot.data.joint_pos
        vel_error = -self.robot.data.joint_vel  # ต้องการ velocity = 0
        
        pd_action = kp * pos_error + kd * vel_error
        
        # รวมกับ action จากการเรียนรู้
        self.actions = pd_action + self.cfg.action_scale * actions.clone()
        
        # จำกัดค่า action
        self.actions = torch.clamp(self.actions, -50.0, 50.0)

    def _apply_action(self) -> None:
        """Apply actions to robot"""
        self.robot.set_joint_effort_target(self.actions)

    def _get_observations(self) -> dict:
        """Get observations"""
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel * 0.1
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        
        # รวม observation
        obs = torch.cat([
            joint_pos[:, :15],
            joint_vel[:, :15],
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Reward for standing upright"""
        # ส่วนสำคัญ: รางวัลการยืน
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        
        # รางวัลความสูง (อยู่สูงจากพื้น)
        height_reward = torch.clamp(root_pos[:, 2] - 0.5, 0.0, 1.0) * 2.0
        
        # รางวัลการยืนตรง (quaternion z-component ใกล้ 1)
        upright_reward = (root_quat[:, 0] ** 2) * 1.0  # w component
        
        # ลงโทษการเอียง
        tilt_penalty = -torch.norm(root_quat[:, 1:3], dim=1) * 0.5  # x,y components
        
        # ลงโทษการเคลื่อนไหวมากเกินไป
        joint_penalty = -0.01 * torch.sum(torch.abs(self.robot.data.joint_pos), dim=1)
        velocity_penalty = -0.01 * torch.sum(torch.abs(self.robot.data.joint_vel), dim=1)
        
        # รวมรางวัล
        total_reward = height_reward + upright_reward + tilt_penalty + joint_penalty + velocity_penalty
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Termination conditions"""
        root_pos = self.robot.data.root_pos_w
        
        # ยุติถ้าล้มลง (ต่ำกว่า 0.3m)
        height_termination = root_pos[:, 2] < 0.3
        
        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        return height_termination, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments"""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        # Reset ในท่ายืน
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 2] = 1.2  # ความสูง 1.2 เมตร
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        
        # Reset joints ในท่ายืน
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)

def test_standing():
    """ทดสอบการยืน"""
    
    print("🚀 แก้ปัญหาหุ่นยนต์นอน - ทำให้ยืนได้!")
    print(f"📊 Environments: {args_cli.num_envs}")
    
    # Create environment
    env_cfg = StandingRobotCfg()
    env = StandingRobotEnv(cfg=env_cfg)
    
    print(f"🤖 Environment created: {env.num_envs} robots")
    print(f"🎮 Action space: {env.action_space}")
    
    print("🏃‍♂️ ทดสอบการยืน...")
    
    # Reset และให้ยืนสักครู่
    obs, _ = env.reset()
    
    for step in range(1000):  # ยืน 1000 steps
        # ส่ง action เป็น 0 (ให้ PD control ทำงาน)
        actions = torch.zeros(env.num_envs, 21, device=env.device)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if step % 100 == 0:
            avg_reward = reward.mean().item()
            avg_height = env.robot.data.root_pos_w[:, 2].mean().item()
            print(f"Step {step:4d}: Reward = {avg_reward:.3f}, Height = {avg_height:.3f}m")
            
            if avg_height > 0.8:
                print("✅ หุ่นยนต์ยืนได้แล้ว!")
            else:
                print("⚠️ หุ่นยนต์ยังไม่ค่อยยืน...")
        
        if terminated.any():
            print(f"💥 หุ่นยนต์ล้มที่ step {step}")
            break
    
    print("✅ การทดสอบเสร็จสิ้น!")
    env.close()

def main():
    """Main function"""
    test_standing()

if __name__ == "__main__":
    main()
    simulation_app.close()