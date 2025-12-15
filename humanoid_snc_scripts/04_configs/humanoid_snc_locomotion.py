# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid SNC locomotion environment using IsaacLab."""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import your robot configuration
try:
    from source.isaaclab_assets.isaaclab_assets.robots.humanoid_snc import HUMANOID_SNC_CFG
except ImportError:
    # Fallback if module structure is different
    import sys
    sys.path.append('/home/sncbot/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots')
    from humanoid_snc import HUMANOID_SNC_CFG

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class HumanoidSNCEnvCfg(DirectRLEnvCfg):
    """Configuration for Humanoid SNC locomotion environment."""
    
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 22  # 22 joints for your robot
    observation_space = 75
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
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
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_SNC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Joint gears for your 22 joints - you may need to adjust these values based on your robot
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
        0.0,   # Empty slot (your YAML shows one empty joint)
    ]

    # Locomotion reward parameters
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    joints_at_limit_cost_scale: float = 0.1
    death_cost: float = -1.0
    termination_height: float = 0.31
    alive_reward_scale: float = 0.5


# Create the environment class
class HumanoidSNCEnv(LocomotionEnv):
    """Humanoid SNC locomotion environment."""
    
    cfg: HumanoidSNCEnvCfg

    def __init__(self, cfg: HumanoidSNCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


def main():
    """Main function."""
    import argparse
    from omni.isaac.lab.app import AppLauncher
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Humanoid SNC locomotion environment")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--headless", action="store_true", help="Run headless without GUI")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch Isaac Sim
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import additional modules after launching Isaac Sim
    from isaaclab.utils.dict import print_dict

    # Create environment configuration
    env_cfg = HumanoidSNCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Print the configuration
    print_dict(env_cfg, nesting=4)
    
    # Create environment
    env = HumanoidSNCEnv(cfg=env_cfg)
    print(f"[INFO] Environment created with {env.num_envs} environments")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")

    # Simulate the environment
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Sample random actions
            actions = torch.rand(env.action_space, device=env.device)
            actions = actions.unsqueeze(0).expand(env.num_envs, -1)
            
            # Step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # Reset if needed
            if count % 1000 == 0:
                env.reset()
                print(f"[INFO] Resetting environment at step: {count}")
            
            count += 1

    # Clean up
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()