# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MINIMAL Humanoid SNC R5 environment - just walk forward."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

@configclass
class HumanoidSNCR5MinimalEnvCfg(DirectRLEnvCfg):
    # timing
    episode_length_s = 10.0
    decimation = 2
    action_scale = 1.0

    # SNC_R5 joints (23 DOF)
    num_actions = 23
    num_observations = 48  # MINIMAL observations
    num_states = 0

    # gym spaces
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_SNC_R5_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Joint gears for R5 (23 joints) - Simple uniform
    joint_gears: list[float] = [50.0] * 23  # All joints same gear

    # MINIMAL reward weights 
    forward_reward_scale: float = 1.0      
    alive_reward_scale: float = 1.0        
    
    # Termination conditions
    death_cost: float = -1.0
    termination_height: float = 0.4        
    max_torso_angle: float = 1.0           # Very lenient

    # Movement targets
    target_forward_velocity: float = 0.5   # Slow target
    angular_velocity_scale: float = 0.25   
    action_scale: float = 1.0              


class HumanoidSNCR5MinimalEnv(LocomotionEnv):
    """MINIMAL walking environment - just move forward."""

    cfg: HumanoidSNCR5MinimalEnvCfg

    def __init__(self, cfg: HumanoidSNCR5MinimalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._num_actions = int(self.cfg.num_actions)
        self._num_obs = int(self.cfg.num_observations)

        # Gym spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._num_obs,), dtype=np.float32
        )

        self.num_actions = self._num_actions

        # Joint gears
        self.joint_gears = torch.tensor(
            self.cfg.joint_gears, device=self.device, dtype=torch.float32
        ).view(1, -1)

    def _get_observations(self) -> dict:
        """MINIMAL observations - just basics."""
        # Base velocity and position
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        
        # Joint positions and velocities
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        
        # MINIMAL observations: 3+3+3+4+23+12 = 48
        observations = torch.cat([
            base_lin_vel,               # 3
            base_ang_vel,               # 3  
            base_pos,                   # 3
            base_quat,                  # 4
            dof_pos,                    # 23
            dof_vel[:, :12],           # 12 (only first 12 joint velocities)
        ], dim=-1)
        
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        """SUPER SIMPLE reward - just walk forward."""
        # Base velocity
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_pos = self.robot.data.root_pos_w
        
        # ONLY 2 COMPONENTS:
        # 1. Forward velocity (linear reward)
        forward_velocity = base_lin_vel[:, 0]
        forward_reward = torch.clamp(forward_velocity, min=0.0, max=2.0)  # 0-2 reward for speed
        
        # 2. Stay upright 
        upright_reward = torch.clamp(base_pos[:, 2] - 0.5, min=0.0, max=1.0)  # 0-1 reward for height
        
        # Total reward
        total_reward = forward_reward + 0.5 * upright_reward + self.cfg.alive_reward_scale
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple termination - just height."""
        base_pos = self.robot.data.root_pos_w
        
        # Only height termination
        height_term = base_pos[:, 2] < self.cfg.termination_height
        
        # Time limit termination
        time_outs = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated = height_term
        
        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments."""
        super()._reset_idx(env_ids)