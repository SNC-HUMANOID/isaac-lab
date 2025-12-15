#!/usr/bin/env python3

"""Simple training script for Humanoid SNC using RSL-RL."""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train Humanoid SNC")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Import the locomotion base class
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

# Define your robot config
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
class HumanoidSNCEnvCfg(DirectRLEnvCfg):
    """Configuration for Humanoid SNC environment."""
    
    # Environment settings
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21  # 21 real joints (excluding empty first entry)
    observation_space = 75
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    
    # Terrain
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

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True
    )

    # Robot
    robot: ArticulationCfg = HUMANOID_SNC_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Joint gears for your 21 joints
    joint_gears: list = [
        50.0,  # left_shoulder_pitch_joint
        50.0,  # left_shoulder_roll_joint  
        50.0,  # left_shoulder_yaw_joint
        30.0,  # left_elbow_joint
        50.0,  # right_shoulder_pitch_joint
        50.0,  # right_shoulder_roll_joint
        50.0,  # right_shoulder_yaw_joint
        30.0,  # right_elbow_joint
        67.5,  # Pelvis_joint
        90.0,  # left_hip_pitch_joint
        45.0,  # left_hip_roll_joint
        45.0,  # left_hip_yaw_joint
        22.5,  # left_ankle_pitch_joint
        22.5,  # left_ankle_roll_joint
        22.5,  # left_ankle_yaw_joint
        90.0,  # right_hip_pitch_joint
        45.0,  # right_hip_roll_joint
        45.0,  # right_hip_yaw_joint
        22.5,  # right_ankle_pitch_joint
        22.5,  # right_ankle_roll_joint
        22.5,  # right_ankle_yaw_joint
    ]

    # Reward parameters
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    joints_at_limit_cost_scale: float = 0.1
    death_cost: float = -1.0
    termination_height: float = 0.31
    alive_reward_scale: float = 0.5
    
    # Additional required reward parameters
    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class HumanoidSNCEnv(LocomotionEnv):
    """Humanoid SNC locomotion environment."""
    cfg: HumanoidSNCEnvCfg

    def __init__(self, cfg: HumanoidSNCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


def main():
    """Main training function."""
    
    # Create environment
    env_cfg = HumanoidSNCEnvCfg()
    env = HumanoidSNCEnv(cfg=env_cfg)
    
    print(f"[INFO] Environment created with {env.num_envs} environments")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")

    # Run simple test
    count = 0
    while simulation_app.is_running() and count < 1000:
        with torch.inference_mode():
            # Random actions
            actions = torch.rand((env.num_envs, env.action_space.shape[-1]), device=env.device)
            
            # Step environment
            obs, rew, terminated, truncated, info = env.step(actions)
            
            if count % 100 == 0:
                print(f"[INFO] Step: {count}, Mean reward: {rew.mean():.3f}")
                
            if count % 500 == 0:
                env.reset()
            
            count += 1

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()