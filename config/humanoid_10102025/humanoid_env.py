# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab_assets.robots.humanoid_10102025 import HUMANOID_10102025_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    # timing
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0

    # Humanoid 10102025 has 21 actuated joints (including pelvis_joint)
    num_actions = 21
    num_observations = 75

    # Gym spaces
    observation_space = 75
    action_space = 21

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_10102025_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Joint gears for 21 DOF (12 leg + 1 pelvis + 8 arm joints)
    joint_gears: list[float] = [
        # Left leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        67.5, 67.5, 67.5, 45.0, 22.5, 22.5,
        # Right leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        67.5, 67.5, 67.5, 45.0, 22.5, 22.5,
        # Pelvis (1)
        67.5,
        # Left arm (4): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
        45.0, 45.0, 45.0, 22.5,
        # Right arm (4): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
        45.0, 45.0, 45.0, 22.5,
    ]

    # reward weights
    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -10.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class HumanoidEnv(LocomotionEnv):
    """Locomotion environment for Humanoid 10102025 robot."""

    cfg: HumanoidEnvCfg
