#!/usr/bin/env python3
"""4️⃣ การเคลื่อนไหวแบบง่ายๆ - ไม่ใช้ RL"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple Humanoid SNC Movement")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

# Robot config
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
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=300.0,
            velocity_limit=10.0,
            stiffness=80.0,
            damping=4.0,
        ),
    },
)

@configclass
class SimpleMovementEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 75
    state_space = 0
    
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average", 
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True
    )
    robot: ArticulationCfg = HUMANOID_SNC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Required for locomotion env
    joint_gears: list = [50.0] * 21
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    joints_at_limit_cost_scale: float = 0.1
    death_cost: float = -1.0
    termination_height: float = 0.31
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

class SimpleMovementEnv(LocomotionEnv):
    cfg: SimpleMovementEnvCfg
    def __init__(self, cfg: SimpleMovementEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

def generate_walking_pattern(step, num_joints=21):
    """สร้าง pattern การเดินแบบง่ายๆ"""
    t = step * 0.02  # เวลา
    
    actions = torch.zeros(num_joints)
    
    # Hip joints (เดิน)
    hip_swing = 0.3 * math.sin(t)
    actions[9] = hip_swing   # left_hip_pitch
    actions[15] = -hip_swing # right_hip_pitch (ตรงข้าม)
    
    # Ankle joints (เสถียรภาพ)
    ankle_balance = 0.1 * math.sin(t * 2)
    actions[12] = ankle_balance  # left_ankle_pitch
    actions[18] = -ankle_balance # right_ankle_pitch
    
    # Arms (แกว่ง)
    arm_swing = 0.2 * math.sin(t * 1.5)
    actions[0] = arm_swing   # left_shoulder_pitch
    actions[4] = -arm_swing  # right_shoulder_pitch
    
    return actions

def main():
    # Create environment
    env_cfg = SimpleMovementEnvCfg()
    env = SimpleMovementEnv(cfg=env_cfg)
    
    print("🤖 Simple Humanoid Movement")
    print(f"📊 Environments: {env.num_envs}")
    print(f"🎮 Action space: {env.action_space}")
    print("🚶‍♂️ Generating walking pattern...")
    
    step = 0
    try:
        while simulation_app.is_running() and step < 2000:
            with torch.inference_mode():
                # Generate walking pattern
                base_actions = generate_walking_pattern(step)
                
                # Apply to all environments
                actions = base_actions.unsqueeze(0).expand(env.num_envs, -1).to(env.device)
                
                # Add small random noise
                noise = torch.randn_like(actions) * 0.01
                actions = actions + noise
                
                # Step environment
                obs, rew, terminated, truncated, info = env.step(actions)
                
                if step % 100 == 0:
                    # Get robot position from the environment directly
                    robot_pos = env.scene["robot"].data.root_pos_w
                    avg_height = robot_pos[:, 2].mean()  # Z position
                    print(f"⏱️  Step: {step:4d}, Height: {avg_height:.2f}m, Reward: {rew.mean():.3f}")
                
                # Reset if robots fall
                if step % 500 == 0:
                    env.reset()
                    print("🔄 Reset environments")
                
                step += 1
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    print("✨ Completed!")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()