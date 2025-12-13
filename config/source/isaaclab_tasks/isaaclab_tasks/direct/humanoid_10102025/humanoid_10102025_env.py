# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid 10102025 environment for reinforcement learning."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
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
class Humanoid10102025EnvCfg(DirectRLEnvCfg):
    # timing
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0

    # Humanoid 10102025 joints
    num_actions = 20  # 10102025 has 20 joints (12 leg + 8 arm)
    num_observations = 72  # Adjusted for 10102025 (20*3 + 12 base obs)
    num_states = 0

    # gym spaces
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(72,), dtype=np.float32)
    state_space = 0  # No privileged states

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

    # Joint gears for Humanoid 10102025 (20 joints)
    # Order: left_leg (6), right_leg (6), left_arm (4), right_arm (4)
    joint_gears: list[float] = [
        # Left leg
        45.0, 45.0, 45.0,  # hip_pitch, hip_roll, hip_yaw
        90.0,              # knee
        22.5, 22.5,        # ankle_pitch, ankle_roll
        # Right leg
        45.0, 45.0, 45.0,  # hip_pitch, hip_roll, hip_yaw
        90.0,              # knee
        22.5, 22.5,        # ankle_pitch, ankle_roll
        # Left arm
        45.0, 45.0, 45.0,  # shoulder_pitch, shoulder_roll, shoulder_yaw
        22.5,              # elbow
        # Right arm
        45.0, 45.0, 45.0,  # shoulder_pitch, shoulder_roll, shoulder_yaw
        22.5,              # elbow
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


class Humanoid10102025Env(LocomotionEnv):
    """Environment for Humanoid 10102025 robot."""

    cfg: Humanoid10102025EnvCfg

    def __init__(self, cfg: Humanoid10102025EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._num_actions = int(self.cfg.num_actions)
        self._num_obs = int(self.cfg.num_observations)

        # -------------------- Gym spaces --------------------
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._num_obs,), dtype=np.float32
        )

        # For runner
        self.num_actions = self._num_actions

        # -------------------- joint_gears -> (1, num_actions) --------------------
        assert len(self.cfg.joint_gears) == self._num_actions, \
            f"joint_gears ({len(self.cfg.joint_gears)}) != num_actions ({self._num_actions})"

        self.joint_gears = torch.tensor(
            self.cfg.joint_gears, device=self.device, dtype=torch.float32
        ).view(1, -1)  # (1, num_actions)

        print(f"[INFO] Humanoid10102025Env initialized successfully!")
        print(f"[INFO]   - Number of actions: {self._num_actions}")
        print(f"[INFO]   - Number of observations: {self._num_obs}")
        print(f"[INFO]   - Number of environments: {self.num_envs}")


# Register the environment with gymnasium
gym.register(
    id="Isaac-Humanoid-10102025-Direct-v0",
    entry_point="isaaclab_tasks.direct.humanoid_10102025.humanoid_10102025_env:Humanoid10102025Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Humanoid10102025EnvCfg,
    },
)
