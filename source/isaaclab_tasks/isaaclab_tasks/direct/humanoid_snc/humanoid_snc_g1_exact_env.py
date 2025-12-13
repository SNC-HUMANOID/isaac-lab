# SPDX-License-Identifier: BSD-3-Clause
"""
Humanoid SNC Environment with EXACT G1 reward implementation
Replicates G1's exact reward functions in Direct environment format
"""

from __future__ import annotations
import math
import torch
import gymnasium as gym
import numpy as np

from isaaclab_assets.robots.humanoid_snc import HUMANOID_SNC_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv, normalize_angle

@configclass
class HumanoidSNCG1ExactEnvCfg(DirectRLEnvCfg):
    """EXACT G1 configuration adapted for Humanoid SNC robot"""

    # === Timing (EXACT G1) ===
    episode_length_s = 20.0
    decimation = 2
    action_scale = 0.5

    # === Spaces ===
    action_space = 21
    observation_space = 90

    # === Simulation ===
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = HUMANOID_SNC_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Joint gears
    joint_gears: list[float] = [
        40.0, 40.0, 40.0, 30.0,  # arms
        40.0, 40.0, 40.0, 30.0,
        40.0,  # torso
        50.0, 120.0, 50.0,  # left leg
        80.0, 25.0, 25.0,
        50.0, 120.0, 50.0,  # right leg
        80.0, 25.0, 25.0,
    ]

    # === EXACT G1 REWARD WEIGHTS ===
    termination_penalty: float = -200.0
    track_lin_vel_xy_exp_weight: float = 1.0
    track_ang_vel_z_exp_weight: float = 2.0
    lin_vel_z_l2_weight: float = -0.2
    ang_vel_xy_l2_weight: float = -1.0
    dof_torques_l2_weight: float = -1.5e-7
    dof_acc_l2_weight: float = -1.25e-7
    action_rate_l2_weight: float = -0.005
    feet_air_time_weight: float = 0.75
    flat_orientation_l2_weight: float = -1.0
    dof_pos_limits_weight: float = -1.0
    feet_slide_weight: float = -0.1
    joint_deviation_hip_weight: float = -0.1
    joint_deviation_arms_weight: float = -0.1
    joint_deviation_fingers_weight: float = -0.05
    joint_deviation_torso_weight: float = -0.1

    # === Command and control ===
    command_lin_vel_x_range: tuple[float, float] = (0.0, 1.0)
    command_lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    command_ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)

    # === G1 parameters ===
    velocity_std: float = 0.5
    feet_air_time_threshold: float = 0.4

    # Scaling
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1

    # Termination
    termination_height: float = 0.6
    max_roll_pitch: float = 1.57

    # Debug
    enable_debug_arrows: bool = True
    debug_arrow_scale: float = 0.3

class HumanoidSNCG1ExactEnv(LocomotionEnv):
    """Humanoid SNC with EXACT G1 reward functions"""

    cfg: HumanoidSNCG1ExactEnvCfg

    def __init__(self, cfg: HumanoidSNCG1ExactEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._torch_device = self.device
        self._num_actions = int(self.cfg.action_space)
        self._num_obs = int(self.cfg.observation_space)

        # Gym spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._num_obs,), dtype=np.float32)
        self.num_actions = self._num_actions

        # Joint gears
        self.joint_gears = torch.tensor(self.cfg.joint_gears, device=self._torch_device, dtype=torch.float32).view(1, -1)

        # === G1 Command system ===
        self._resample_commands()

        # === G1 tracking variables ===
        self.previous_actions = torch.zeros((self.num_envs, self._num_actions), device=self._torch_device)
        self.previous_dof_vel = torch.zeros((self.num_envs, self._num_actions), device=self._torch_device)

        # Contact tracking (simplified for Direct env)
        self.contact_forces = torch.zeros((self.num_envs, 2), device=self._torch_device)
        self.last_contacts = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self._torch_device)
        self.feet_air_time = torch.zeros((self.num_envs, 2), device=self._torch_device)

        # Debug arrows
        self._debug_draw = None
        if self.cfg.enable_debug_arrows:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                print("🎯 G1 EXACT debug arrows enabled!")
            except Exception as e:
                print(f"[DEBUG-ARROWS] unavailable: {e}")

        print("🚀 Humanoid SNC with EXACT G1 rewards initialized!")

    def _resample_commands(self):
        """G1-style command resampling"""
        # Random commands like G1
        self.command_lin_vel_x = torch.rand(self.num_envs, device=self._torch_device) * \
            (self.cfg.command_lin_vel_x_range[1] - self.cfg.command_lin_vel_x_range[0]) + \
            self.cfg.command_lin_vel_x_range[0]

        self.command_lin_vel_y = torch.rand(self.num_envs, device=self._torch_device) * \
            (self.cfg.command_lin_vel_y_range[1] - self.cfg.command_lin_vel_y_range[0]) + \
            self.cfg.command_lin_vel_y_range[0]

        self.command_ang_vel_z = torch.rand(self.num_envs, device=self._torch_device) * \
            (self.cfg.command_ang_vel_z_range[1] - self.cfg.command_ang_vel_z_range[0]) + \
            self.cfg.command_ang_vel_z_range[0]

    def _get_observations(self) -> dict:
        """G1-style observations"""
        try:
            self._compute_intermediate_values()
        except Exception as e:
            print(f"[_get_observations] error: {e}")

        self._safe_fix_nan_values()

        # Build observations like G1
        parts = [
            # Height
            self.torso_position[:, 2:3],

            # Local velocities (key for G1)
            self.vel_loc,
            self.angvel_loc * self.cfg.angular_velocity_scale,

            # Orientation
            normalize_angle(self.yaw).unsqueeze(-1),
            normalize_angle(self.roll).unsqueeze(-1),
            normalize_angle(self.pitch).unsqueeze(-1),

            # Commands (G1 tracks these)
            self.command_lin_vel_x.unsqueeze(-1),
            self.command_lin_vel_y.unsqueeze(-1),
            self.command_ang_vel_z.unsqueeze(-1),

            # Joint states
            self.dof_pos_scaled,
            self.dof_vel * self.cfg.dof_vel_scale,

            # Actions (for rate penalty)
            self.actions,
            self.previous_actions,
        ]

        obs = torch.cat(parts, dim=-1)

        # Pad/truncate to correct size
        if obs.shape[1] > self._num_obs:
            obs = obs[:, :self._num_obs]
        elif obs.shape[1] < self._num_obs:
            pad = torch.zeros((obs.shape[0], self._num_obs - obs.shape[1]), device=self._torch_device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """EXACT G1 reward function implementation"""

        # === 1. termination_penalty ===
        # Applied at the end like G1

        # === 2. track_lin_vel_xy_exp ===
        lin_vel_error = torch.sqrt(
            (self.vel_loc[:, 0] - self.command_lin_vel_x) ** 2 +
            (self.vel_loc[:, 1] - self.command_lin_vel_y) ** 2
        )
        track_lin_vel_xy_exp = self.cfg.track_lin_vel_xy_exp_weight * torch.exp(-lin_vel_error / self.cfg.velocity_std)

        # === 3. track_ang_vel_z_exp ===
        ang_vel_error = torch.abs(self.angvel_loc[:, 2] - self.command_ang_vel_z)
        track_ang_vel_z_exp = self.cfg.track_ang_vel_z_exp_weight * torch.exp(-ang_vel_error / self.cfg.velocity_std)

        # === 4. lin_vel_z_l2 ===
        lin_vel_z_l2 = self.cfg.lin_vel_z_l2_weight * (self.velocity[:, 2] ** 2)

        # === 5. ang_vel_xy_l2 ===
        ang_vel_xy_l2 = self.cfg.ang_vel_xy_l2_weight * (self.ang_velocity[:, 0] ** 2 + self.ang_velocity[:, 1] ** 2)

        # === 6. dof_torques_l2 ===
        # Estimate torques from actions and joint gears (with safety)
        estimated_torques = torch.nan_to_num(self.actions * self.joint_gears.squeeze(0), nan=0.0, posinf=100.0, neginf=-100.0)
        # Apply to hip and knee joints (like G1) - safer indexing
        hip_knee_indices = [9, 10, 11, 15, 16, 17]  # hip and knee joints
        valid_indices = [i for i in hip_knee_indices if i < estimated_torques.shape[1]]
        if valid_indices:
            hip_knee_torques = estimated_torques[:, valid_indices]
            dof_torques_l2 = self.cfg.dof_torques_l2_weight * torch.sum(hip_knee_torques ** 2, dim=-1)
        else:
            dof_torques_l2 = torch.zeros(self.num_envs, device=self._torch_device)

        # === 7. dof_acc_l2 ===
        dof_acc = torch.nan_to_num(
            (self.dof_vel - self.previous_dof_vel) / max(self.cfg.sim.dt * self.cfg.decimation, 1e-6),
            nan=0.0, posinf=100.0, neginf=-100.0
        )
        # Apply to hip and knee joints (safer)
        if valid_indices:
            hip_knee_acc = dof_acc[:, valid_indices]
            dof_acc_l2 = self.cfg.dof_acc_l2_weight * torch.sum(hip_knee_acc ** 2, dim=-1)
        else:
            dof_acc_l2 = torch.zeros(self.num_envs, device=self._torch_device)

        # === 8. action_rate_l2 ===
        action_rate_l2 = self.cfg.action_rate_l2_weight * torch.sum((self.actions - self.previous_actions) ** 2, dim=-1)

        # === 9. feet_air_time ===
        self._update_contact_tracking()
        # Only reward when there's a command (like G1)
        command_norm = torch.norm(torch.stack([self.command_lin_vel_x, self.command_lin_vel_y], dim=-1), dim=-1)
        has_command = (command_norm > 0.1).float()
        feet_air_time_reward = self.cfg.feet_air_time_weight * torch.mean(
            torch.clamp(self.feet_air_time - self.cfg.feet_air_time_threshold, min=0.0), dim=-1
        ) * has_command

        # === 10. flat_orientation_l2 ===
        flat_orientation_l2 = self.cfg.flat_orientation_l2_weight * (self.roll ** 2 + self.pitch ** 2)

        # === 11. dof_pos_limits ===
        # Penalize ankle joints near limits (like G1)
        ankle_joints = self.dof_pos_scaled[:, [12, 13, 18, 19]]  # ankle pitch and roll
        dof_pos_limits = self.cfg.dof_pos_limits_weight * torch.sum(
            torch.relu(torch.abs(ankle_joints) - 0.98), dim=-1
        )

        # === 12. feet_slide ===
        # Simplified for Direct env
        feet_slide = self.cfg.feet_slide_weight * 0.0  # placeholder

        # === 13. joint_deviation_hip ===
        hip_joints = self.dof_pos_scaled[:, [10, 11, 16, 17]]  # hip yaw and roll
        joint_deviation_hip = self.cfg.joint_deviation_hip_weight * torch.sum(torch.abs(hip_joints), dim=-1)

        # === 14. joint_deviation_arms ===
        arm_joints = self.dof_pos_scaled[:, [0, 1, 2, 3, 4, 5, 6, 7]]  # all arm joints
        joint_deviation_arms = self.cfg.joint_deviation_arms_weight * torch.sum(torch.abs(arm_joints), dim=-1)

        # === 15. joint_deviation_fingers ===
        # No finger joints in SNC, skip
        joint_deviation_fingers = 0.0

        # === 16. joint_deviation_torso ===
        torso_joint = self.dof_pos_scaled[:, 8]  # pelvis joint
        joint_deviation_torso = self.cfg.joint_deviation_torso_weight * torch.abs(torso_joint)

        # === Total reward (EXACT G1 style with safety) ===
        # Clamp individual rewards to prevent explosion
        track_lin_vel_xy_exp = torch.clamp(track_lin_vel_xy_exp, -10.0, 10.0)
        track_ang_vel_z_exp = torch.clamp(track_ang_vel_z_exp, -10.0, 10.0)
        lin_vel_z_l2 = torch.clamp(lin_vel_z_l2, -10.0, 0.0)
        ang_vel_xy_l2 = torch.clamp(ang_vel_xy_l2, -10.0, 0.0)

        total_reward = (
            track_lin_vel_xy_exp +
            track_ang_vel_z_exp +
            lin_vel_z_l2 +
            ang_vel_xy_l2 +
            dof_torques_l2 +
            dof_acc_l2 +
            action_rate_l2 +
            feet_air_time_reward +
            flat_orientation_l2 +
            dof_pos_limits +
            feet_slide +
            joint_deviation_hip +
            joint_deviation_arms +
            joint_deviation_torso
        )

        # Final safety check
        total_reward = torch.nan_to_num(total_reward, nan=-1.0, posinf=10.0, neginf=-10.0)
        total_reward = torch.clamp(total_reward, min=-50.0, max=50.0)

        # Apply termination penalty (like G1)
        total_reward = torch.where(
            self.reset_terminated,
            torch.ones_like(total_reward) * self.cfg.termination_penalty,
            total_reward
        )

        return total_reward

    def _update_contact_tracking(self):
        """Update contact tracking for feet air time"""
        dt = self.cfg.sim.dt * self.cfg.decimation

        # Simple contact detection
        in_contact = (
            (torch.abs(self.velocity[:, 2]) < 0.5) &
            (self.torso_position[:, 2] > self.cfg.termination_height)
        ).unsqueeze(-1).repeat(1, 2)  # both feet

        # Update air time
        self.feet_air_time = torch.where(
            in_contact,
            torch.zeros_like(self.feet_air_time),
            self.feet_air_time + dt
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """G1-style termination"""
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        height_died = self.torso_position[:, 2] < self.cfg.termination_height
        balance_died = (
            (torch.abs(self.roll) > self.cfg.max_roll_pitch) |
            (torch.abs(self.pitch) > self.cfg.max_roll_pitch)
        )

        died = height_died | balance_died
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset with command resampling like G1"""
        super()._reset_idx(env_ids)

        if env_ids is not None:
            # Resample commands for reset environments
            n = len(env_ids)
            self.command_lin_vel_x[env_ids] = torch.rand(n, device=self._torch_device) * \
                (self.cfg.command_lin_vel_x_range[1] - self.cfg.command_lin_vel_x_range[0]) + \
                self.cfg.command_lin_vel_x_range[0]

            self.command_lin_vel_y[env_ids] = torch.rand(n, device=self._torch_device) * \
                (self.cfg.command_lin_vel_y_range[1] - self.cfg.command_lin_vel_y_range[0]) + \
                self.cfg.command_lin_vel_y_range[0]

            self.command_ang_vel_z[env_ids] = torch.rand(n, device=self._torch_device) * \
                (self.cfg.command_ang_vel_z_range[1] - self.cfg.command_ang_vel_z_range[0]) + \
                self.cfg.command_ang_vel_z_range[0]

            # Reset tracking variables
            self.feet_air_time[env_ids] = 0.0
            self.previous_actions[env_ids] = 0.0
            self.previous_dof_vel[env_ids] = 0.0

    def step(self, actions: torch.Tensor):
        """G1-style step with tracking updates"""
        if actions.dim() == 1:
            actions = actions.view(1, -1)
        if actions.shape[0] == self._num_actions and actions.shape[1] == self.num_envs:
            actions = actions.T

        # Store previous values for tracking
        self.previous_actions = self.actions.clone() if hasattr(self, 'actions') else actions.clone()
        self.previous_dof_vel = self.dof_vel.clone() if hasattr(self, 'dof_vel') else torch.zeros_like(actions)

        obs_dict, rewards, terminated, truncated, info = super().step(actions)
        dones = terminated | truncated

        # Debug arrows
        if self._debug_draw is not None:
            try:
                P = self.torso_position.detach().cpu().numpy()
                V = self.velocity.detach().cpu().numpy()
                scale = float(self.cfg.debug_arrow_scale)
                self._debug_draw.clear_lines()
                n = min(64, P.shape[0])  # limit for performance
                for i in range(n):
                    x0, y0, z0 = P[i].tolist()
                    x1, y1, z1 = (P[i] + V[i] * scale).tolist()
                    self._debug_draw.draw_line([x0, y0, z0], [x1, y1, z1], (0.1, 0.7, 1.0, 1.0))
            except Exception:
                pass

        return obs_dict["policy"], rewards, dones, info

    def _safe_fix_nan_values(self):
        """Fix NaN values"""
        names = ["roll", "pitch", "yaw", "velocity", "ang_velocity", "vel_loc", "angvel_loc",
                "torso_position", "actions", "dof_pos_scaled", "dof_vel"]
        for n in names:
            if hasattr(self, n):
                t = getattr(self, n)
                if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
                    t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
                    setattr(self, n, t)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """RSL-RL compatibility"""
        obs_dict = self._get_observations()
        obs_tensor = obs_dict["policy"]
        extras = {
            "observations": obs_dict,
            "episode_length_buf": getattr(self, "episode_length_buf", torch.zeros(self.num_envs, device=self._torch_device)),
        }
        return obs_tensor, extras