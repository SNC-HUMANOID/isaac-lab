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
    # ===============================
    # Timing / RL Config
    # ===============================
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0

    # ===============================
    # Observation & Action Spaces
    # ===============================
    num_actions = 21
    num_observations = 75
    observation_space = 75
    action_space = 21

    # ===============================
    # Simulation Parameters
    # ===============================
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

    # ===============================
    # Scene Setup
    # ===============================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # ===============================
    # Robot (Humanoid 10102025)
    # ===============================
    robot: ArticulationCfg = HUMANOID_10102025_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ===============================
    # Motor Gear Ratios
    # ===============================
    joint_gears: list[float] = [
        # Order should match USD joint order
        67.5, 67.5, 67.5, 45.0, 22.5, 22.5,    # Right leg
        67.5, 67.5, 67.5, 45.0, 22.5, 22.5,    # Left leg
        67.5,                                  # Pelvis
        45.0, 45.0, 45.0, 22.5,                # Right arm
        45.0, 45.0, 45.0, 22.5,                # Left arm
    ]

    # ===============================
    # Reward Configuration
    # ===============================
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

    def __init__(self, cfg: HumanoidEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # ⚙️ Ensure joint names and mapping are consistent
        joint_names = self.robot.data.joint_names
        print(f"[DEBUG] Joints loaded ({len(joint_names)}): {joint_names}")

        if len(self.cfg.joint_gears) != len(joint_names):
            print("[WARNING] joint_gears length mismatch! Adjusting automatically.")
            self.cfg.joint_gears = [45.0 for _ in range(len(joint_names))]

        # Store gear tensor
        self.joint_gears = torch.tensor(self.cfg.joint_gears, device=self.device)

    def pre_physics_step(self, actions: torch.Tensor):
        """Applies scaled actions to actuators."""
        # Normalize actions (-1 to 1) and scale by gear ratio
        torques = actions * self.cfg.action_scale * self.joint_gears
        self.robot.set_joint_efforts(torques)

    def post_physics_step(self):
        """Custom post-step logic."""
        super().post_physics_step()

        # Stabilize torso uprightness
        pelvis_height = self.robot.data.root_pos_w[:, 2]
        upright = (pelvis_height > self.cfg.termination_height).float()
        if torch.any(upright == 0):
            self.done = True

