# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Humanoid SNC R5 walking environment with natural gait and arm movement."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import math

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
class HumanoidSNCR5WalkingEnvCfg(DirectRLEnvCfg):
    # timing
    episode_length_s = 20.0
    decimation = 2
    action_scale = 1.0

    # SNC_R5 joints (23 DOF)
    num_actions = 23
    num_observations = 87  # Extended observations for walking
    num_states = 0

    # gym spaces
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(87,), dtype=np.float32)
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_SNC_R5_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Joint gears for R5 (23 joints) - Adjusted for walking
    joint_gears: list[float] = [
        67.5, 67.5, 67.5, 45.0, 45.0,  # Left arm
        67.5, 67.5, 67.5, 45.0, 45.0,  # Right arm  
        67.5,                           # Pelvis
        90.0, 135.0, 67.5, 90.0,       # Left leg (increased hip strength)
        45.0, 45.0,                    # Left ankle
        90.0, 135.0, 67.5, 90.0,       # Right leg (increased hip strength)
        45.0, 45.0,                    # Right ankle
    ]

    # Walking-specific reward weights (adjusted)
    forward_reward_scale: float = 1.2      # Moderate forward movement
    heading_weight: float = 0.3            # Less strict on heading
    up_weight: float = 0.4                 # Higher balance priority
    energy_cost_scale: float = 0.05        # Higher energy penalty
    actions_cost_scale: float = 0.01       # Higher action penalty for smoothness
    alive_reward_scale: float = 0.8        # Reduced base reward
    dof_vel_scale: float = 0.08            # Higher joint velocity penalty
    
    # Walking specific rewards (adjusted)
    gait_cycle_reward: float = 0.4         # Moderate gait pattern reward
    foot_contact_reward: float = 0.3       # Reward proper foot contact
    arm_swing_reward: float = 0.15         # Reduced arm swing reward
    torso_stability_reward: float = 0.6    # Higher torso stability
    
    # Termination conditions
    death_cost: float = -5.0
    termination_height: float = 0.5        # Lower termination - allow robot to get closer to ground
    max_still_time: float = 5.0            # Terminate if standing still too long
    max_torso_angle: float = 0.6           # Stricter torso angle limit (radians)

    # Movement targets (more conservative)
    target_forward_velocity: float = 1.0   # Normal walking speed for longer steps (m/s)
    target_lateral_velocity: float = 0.0   # No lateral movement
    target_angular_velocity: float = 0.0   # No turning

    # Contact detection
    contact_force_scale: float = 0.01
    angular_velocity_scale: float = 0.25    # Required by locomotion env
    action_scale: float = 1.0               # Required by locomotion env


class HumanoidSNCR5WalkingEnv(LocomotionEnv):
    """Environment for natural walking with SNC R5 humanoid."""

    cfg: HumanoidSNCR5WalkingEnvCfg

    def __init__(self, cfg: HumanoidSNCR5WalkingEnvCfg, render_mode: str | None = None, **kwargs):
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
        assert len(self.cfg.joint_gears) == self._num_actions, \
            f"joint_gears ({len(self.cfg.joint_gears)}) != num_actions ({self._num_actions})"

        self.joint_gears = torch.tensor(
            self.cfg.joint_gears, device=self.device, dtype=torch.float32
        ).view(1, -1)

        # Walking-specific variables
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.step_count = torch.zeros(self.num_envs, device=self.device)
        self.still_time = torch.zeros(self.num_envs, device=self.device)  # Track how long standing still
        
        # Target velocities for each environment
        self.target_velocities = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_velocities[:, 0] = self.cfg.target_forward_velocity  # Forward velocity

    def _get_observations(self) -> dict:
        """Compute observations with additional walking-specific features."""
        # Base observations from locomotion env
        obs = super()._get_observations()
        
        # Add walking-specific observations
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        
        # Gait phase (normalized)
        gait_phase_norm = (self.gait_phase / (2 * math.pi)).unsqueeze(-1)
        
        # Target velocity
        target_vel = self.target_velocities
        
        # Joint positions and velocities
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        
        # Foot contact information (estimated from joint positions)
        left_ankle_pos = dof_pos[:, -4]   # left_ankle_roll
        right_ankle_pos = dof_pos[:, -2]  # right_ankle_roll
        foot_contact_est = torch.stack([left_ankle_pos, right_ankle_pos], dim=-1)
        
        # Combine all observations
        observations = torch.cat([
            obs["policy"],           # Base locomotion observations
            gait_phase_norm,         # Gait phase [1]
            target_vel,              # Target velocities [3]
            foot_contact_est,        # Foot contact estimation [2]
            base_ang_vel,            # Angular velocity [3]
        ], dim=-1)
        
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for natural walking with improved posture and step length."""
        # Base velocity and position
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        
        # Joint states
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        
        # Actions
        actions = self.actions
        
        # 1. SIMPLE: Direct forward movement reward
        forward_velocity = base_lin_vel[:, 0]
        forward_reward = torch.clamp(forward_velocity, min=0.0, max=1.5)  # Simple linear reward for moving forward
        
        # 2. SUPER STRONG lateral velocity penalty (PREVENT sideways movement completely)
        lateral_penalty = torch.abs(base_lin_vel[:, 1]) * 5.0  # 5x penalty for ANY sideways drift
        
        # 3. Angular velocity penalty (minimize rotation)
        angular_penalty = torch.sum(torch.abs(base_ang_vel), dim=-1)
        
        # 4. IMPROVED: Torso orientation reward (prevent backward lean)
        # Extract pitch angle from quaternion (forward/backward tilt)
        pitch = torch.atan2(2.0 * (base_quat[:, 3] * base_quat[:, 1] - base_quat[:, 2] * base_quat[:, 0]),
                           1.0 - 2.0 * (base_quat[:, 1]**2 + base_quat[:, 2]**2))
        roll = torch.atan2(2.0 * (base_quat[:, 3] * base_quat[:, 0] + base_quat[:, 1] * base_quat[:, 2]),
                          1.0 - 2.0 * (base_quat[:, 0]**2 + base_quat[:, 1]**2))
        
        # ENHANCED: Strong penalty for backward lean AND roll (side lean)
        torso_upright_reward = torch.exp(-torch.abs(pitch) * 3.0) * torch.exp(-torch.abs(roll) * 5.0)  # Stronger roll penalty
        backward_lean_penalty = torch.clamp(-pitch * 5.0, min=0.0, max=2.0)  # Extra penalty for backward lean
        sideways_lean_penalty = torch.abs(roll) * 8.0  # STRONG penalty for leaning to either side
        
        # 5. IMPROVED: Step length control (prevent oversized steps)
        left_hip_pitch = dof_pos[:, 11]   # left hip pitch
        right_hip_pitch = dof_pos[:, 17]  # right hip pitch
        left_knee = dof_pos[:, 14]        # left knee
        right_knee = dof_pos[:, 20]       # right knee
        
        # RELAXED: Allow longer step length
        step_length_penalty = 0.0
        # Hip pitch can be larger for longer steps (-0.5 to 0.5 radians)
        step_length_penalty += torch.sum(torch.clamp(torch.abs(left_hip_pitch) - 0.5, min=0.0)**2)
        step_length_penalty += torch.sum(torch.clamp(torch.abs(right_hip_pitch) - 0.5, min=0.0)**2)
        # Knee can bend more for longer steps (0 to 1.2 radians)
        step_length_penalty += torch.sum(torch.clamp(torch.abs(left_knee) - 1.2, min=0.0)**2)
        step_length_penalty += torch.sum(torch.clamp(torch.abs(right_knee) - 1.2, min=0.0)**2)
        
        # 6. Energy penalty (reduce excessive joint movements)
        energy_penalty = torch.sum(torch.square(dof_vel), dim=-1)
        
        # 7. Action smoothness (reduce jerkiness)
        action_penalty = torch.sum(torch.square(actions), dim=-1)
        
        # 8. SIMPLE: Basic leg movement reward
        self.gait_phase += 0.1  # Normal phase update
        self.gait_phase = torch.remainder(self.gait_phase, 2 * math.pi)
        
        # AGGRESSIVE: Force leg movement
        left_hip_movement = torch.abs(dof_vel[:, 11])   # left hip pitch velocity
        right_hip_movement = torch.abs(dof_vel[:, 17])  # right hip pitch velocity
        left_knee_movement = torch.abs(dof_vel[:, 14])  # left knee velocity
        right_knee_movement = torch.abs(dof_vel[:, 20]) # right knee velocity
        
        total_leg_movement = left_hip_movement + right_hip_movement + left_knee_movement + right_knee_movement
        gait_reward = total_leg_movement * 0.5  # Strong movement reward
        
        # PENALTY for still legs
        stillness_penalty = torch.exp(-total_leg_movement * 5.0) * 3.0  # Big penalty for not moving legs
        
        # 9. IMPROVED: Controlled arm swing (prevent excessive movement)
        left_shoulder_pitch = dof_pos[:, 0]   # left shoulder pitch
        right_shoulder_pitch = dof_pos[:, 5]  # right shoulder pitch
        
        # Moderate arm swing opposite to legs
        natural_arm_swing = (left_shoulder_pitch - right_shoulder_pitch) * torch.sin(self.gait_phase) * 0.1
        arm_swing_reward = torch.tanh(natural_arm_swing)
        
        # Penalty for excessive arm movement
        arm_movement_penalty = torch.sum(torch.abs(dof_pos[:, 0:10]), dim=-1) * 0.1  # All arm joints
        
        # 10. HIGH: Match new starting height  
        target_height = 1.0   # Match new starting height
        height_reward = torch.exp(-torch.abs(base_pos[:, 2] - target_height) * 2.0)  # Gentle height control
        
        # Prevent floating too high
        height_penalty = torch.clamp(base_pos[:, 2] - 1.1, min=0.0) * 1.0  # Penalty for floating above 1.1m
        
        # 11. Foot clearance reward (prevent dragging feet)
        left_ankle_pitch = dof_pos[:, 15]   # left ankle pitch
        right_ankle_pitch = dof_pos[:, 21]  # right ankle pitch
        foot_clearance_reward = torch.exp(-torch.abs(left_ankle_pitch) - torch.abs(right_ankle_pitch))
        
        # 12. NEW: Leg flexibility reward (encourage smooth joint movements)
        left_leg_joints = dof_pos[:, [11, 12, 13, 14, 15, 16]]  # All left leg joints
        right_leg_joints = dof_pos[:, [17, 18, 19, 20, 21, 22]] # All right leg joints
        
        # Reward smooth joint transitions (low velocity)
        left_leg_smooth = torch.exp(-torch.sum(torch.abs(dof_vel[:, [11, 12, 13, 14, 15, 16]]), dim=-1) * 0.1)
        right_leg_smooth = torch.exp(-torch.sum(torch.abs(dof_vel[:, [17, 18, 19, 20, 21, 22]]), dim=-1) * 0.1)
        leg_smoothness_reward = (left_leg_smooth + right_leg_smooth) / 2.0
        
        # 13. NEW: Knee flexibility reward (encourage natural knee bending)
        # Reward moderate knee flexion during gait cycle
        left_knee_flex = torch.sin(self.gait_phase) * left_knee  # Knee should flex with gait
        right_knee_flex = torch.sin(self.gait_phase + math.pi) * right_knee  # Opposite phase
        knee_flexibility_reward = torch.tanh(left_knee_flex + right_knee_flex) * 0.2
        
        # 14. NEW: Hip flexibility reward (encourage hip swing)
        hip_swing = (left_hip_pitch - right_hip_pitch) * torch.cos(self.gait_phase)
        hip_flexibility_reward = torch.tanh(hip_swing * 0.5) * 0.1
        
        # FORCE WALKING: No standing rewards at all
        total_reward = (
            # ONLY walking rewards - no rewards for standing still
            10.0 * forward_reward +            # MASSIVE forward movement reward
            2.0 * gait_reward +                # Strong leg movement reward
            
            # PUNISHMENT for not moving
            -5.0 * torch.exp(-forward_velocity * 3.0) +  # Huge penalty for slow forward speed
            -stillness_penalty +               # Penalty for still legs
            
            # Basic safety only
            -2.0 * sideways_lean_penalty -     # Don't fall sideways
            -lateral_penalty +                 # Don't drift sideways
            
            # TINY balance reward only when moving
            0.1 * torso_upright_reward * torch.clamp(forward_velocity, min=0.0, max=1.0)  # Balance only if moving
        )
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        base_lin_vel = self.robot.data.root_lin_vel_b
        
        # Track still time
        moving = torch.abs(base_lin_vel[:, 0]) > 0.1  # Moving if forward speed > 0.1 m/s
        self.still_time = torch.where(moving, torch.zeros_like(self.still_time), self.still_time + self.cfg.sim.dt)
        
        # Height-based termination
        height_term = base_pos[:, 2] < self.cfg.termination_height
        
        # Angle-based termination (if robot falls over)
        roll = 2 * (base_quat[:, 3] * base_quat[:, 0] + base_quat[:, 1] * base_quat[:, 2])
        pitch = 2 * (base_quat[:, 3] * base_quat[:, 1] - base_quat[:, 2] * base_quat[:, 0])
        angle_term = (torch.abs(roll) > self.cfg.max_torso_angle) | (torch.abs(pitch) > self.cfg.max_torso_angle)
        
        # FORCE WALKING: Terminate if standing still too long
        still_term = self.still_time > self.cfg.max_still_time
        
        # Time limit termination
        time_outs = self.episode_length_buf >= self.max_episode_length - 1
        
        terminated = height_term | angle_term | still_term
        
        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments."""
        super()._reset_idx(env_ids)
        
        # Reset walking-specific variables
        self.gait_phase[env_ids] = 0.0
        self.step_count[env_ids] = 0.0
        self.still_time[env_ids] = 0.0
        
        # Randomize target velocities slightly for variety
        self.target_velocities[env_ids, 0] = self.cfg.target_forward_velocity + torch.randn(
            len(env_ids), device=self.device
        ) * 0.2  # Add some variation in target speed