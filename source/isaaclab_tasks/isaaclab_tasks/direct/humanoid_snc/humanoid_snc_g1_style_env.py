# SPDX-License-Identifier: BSD-3-Clause
"""
Humanoid SNC Environment with G1-style reward system
Combines the best of G1 training with Humanoid SNC robot
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
class HumanoidSNCG1StyleEnvCfg(DirectRLEnvCfg):
    """G1-style configuration adapted for Humanoid SNC robot"""

    # === Timing (G1-style) ===
    episode_length_s = 20.0  # G1 uses longer episodes
    decimation = 2
    action_scale = 0.5  # Start conservative

    # === Action/Observation Space ===
    action_space = 21
    observation_space = 85  # Enhanced with velocity tracking info

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

    # === Joint Configuration ===
    joint_gears: list[float] = [
        40.0, 40.0, 40.0, 30.0,  # arms - reduced for smoothness
        40.0, 40.0, 40.0, 30.0,
        40.0,  # torso
        50.0, 120.0, 50.0,  # left leg
        80.0, 25.0, 25.0,
        50.0, 120.0, 50.0,  # right leg
        80.0, 25.0, 25.0,
    ]

    # === G1-style Reward Weights ===
    # Velocity tracking (primary objectives)
    track_lin_vel_xy_weight: float = 1.0  # G1 uses 1.0
    track_ang_vel_z_weight: float = 2.0   # G1 uses 2.0

    # Gait and contact
    feet_air_time_weight: float = 0.75     # G1 uses 0.75 for flat terrain
    feet_slide_weight: float = -0.1        # G1 uses -0.1

    # Penalties
    termination_penalty: float = -200.0    # G1 uses -200.0
    lin_vel_z_weight: float = -0.2         # Penalize vertical motion
    ang_vel_xy_weight: float = -1.0        # Penalize roll/pitch rotation
    flat_orientation_weight: float = -1.0  # Keep upright

    # Action regularization
    action_rate_weight: float = -0.005     # G1 uses -0.005
    dof_acc_weight: float = -1.25e-7       # G1 uses -1.25e-7
    dof_torques_weight: float = -1.5e-7    # G1 uses -1.5e-7

    # Joint deviation penalties
    joint_deviation_hip_weight: float = -0.1      # G1 uses -0.1
    joint_deviation_arms_weight: float = -0.1     # G1 uses -0.1
    joint_deviation_torso_weight: float = -0.1    # G1 uses -0.1

    # Joint limits
    dof_pos_limits_weight: float = -1.0    # G1 uses -1.0

    # === Velocity Control Parameters (G1-style) ===
    target_lin_vel_x: float = 1.0  # m/s forward speed
    target_lin_vel_y: float = 0.0  # m/s sideways speed
    target_ang_vel_z: float = 0.0  # rad/s turning speed
    velocity_std: float = 0.5      # G1 uses 0.5 for exponential reward

    # === Velocity Scaling ===
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1

    # === Contact Settings ===
    feet_air_time_threshold: float = 0.4   # G1 uses 0.4 seconds
    contact_force_threshold: float = 1.0    # N

    # === Termination ===
    termination_height: float = 0.6
    max_roll_pitch: float = 1.57  # 90 degrees - more lenient

    # === Debug Features ===
    enable_debug_arrows: bool = True
    debug_arrow_scale: float = 0.3
    debug_arrow_max_lines: int = 256
    debug_arrow_color: tuple[float, float, float, float] = (0.1, 0.7, 1.0, 1.0)

class HumanoidSNCG1StyleEnv(LocomotionEnv):
    """Humanoid SNC with G1-style reward system and velocity tracking"""

    cfg: HumanoidSNCG1StyleEnvCfg

    def __init__(self, cfg: HumanoidSNCG1StyleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Device setup
        sim_dev = self.sim.device if hasattr(self, "sim") else ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = torch.device(sim_dev) if isinstance(sim_dev, str) else sim_dev

        self._num_actions = int(self.cfg.action_space)
        self._num_obs = int(self.cfg.observation_space)

        # Gym spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._num_obs,), dtype=np.float32)
        self.num_actions = self._num_actions

        # Joint gears
        assert len(self.cfg.joint_gears) == self._num_actions
        self.joint_gears = torch.tensor(self.cfg.joint_gears, device=self._torch_device, dtype=torch.float32).view(1, -1)

        # G1-style velocity commands (fixed for now, can be randomized later)
        self.command_lin_vel_x = torch.full((self.num_envs,), self.cfg.target_lin_vel_x, device=self._torch_device)
        self.command_lin_vel_y = torch.full((self.num_envs,), self.cfg.target_lin_vel_y, device=self._torch_device)
        self.command_ang_vel_z = torch.full((self.num_envs,), self.cfg.target_ang_vel_z, device=self._torch_device)

        # Tracking previous actions for action rate penalty
        self.previous_actions = torch.zeros((self.num_envs, self._num_actions), device=self._torch_device)

        # Contact tracking for feet air time
        self.contact_forces = torch.zeros((self.num_envs, 2), device=self._torch_device)  # left, right foot
        self.last_contacts = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self._torch_device)
        self.feet_air_time = torch.zeros((self.num_envs, 2), device=self._torch_device)

        # Debug arrows
        self._debug_draw = None
        if self.cfg.enable_debug_arrows:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                print("🎯 G1-style debug arrows enabled!")
            except Exception as e:
                print(f"[DEBUG-ARROWS] unavailable: {e}")

        print("🚀 Humanoid SNC with G1-style rewards initialized!")

    def _get_observations(self) -> dict:
        """G1-style observations with velocity tracking info"""
        try:
            self._compute_intermediate_values()
        except Exception as e:
            print(f"[_get_observations] compute_intermediate_values error: {e}")

        self._safe_fix_nan_values()

        # Base observations
        parts = [
            # Position and orientation
            self.torso_position[:, 2:3],  # height
            normalize_angle(self.roll).unsqueeze(-1),
            normalize_angle(self.pitch).unsqueeze(-1),
            normalize_angle(self.yaw).unsqueeze(-1),

            # Velocities (key for G1-style control)
            self.vel_loc,  # local linear velocity (3D)
            self.angvel_loc * self.cfg.angular_velocity_scale,  # local angular velocity (3D)

            # Velocity commands (for tracking)
            self.command_lin_vel_x.unsqueeze(-1),
            self.command_lin_vel_y.unsqueeze(-1),
            self.command_ang_vel_z.unsqueeze(-1),

            # Joint states
            self.dof_pos_scaled,  # normalized joint positions
            self.dof_vel * self.cfg.dof_vel_scale,  # joint velocities

            # Actions (for temporal consistency)
            self.actions,

            # Previous actions (for action rate calculation)
            self.previous_actions,
        ]

        obs = torch.cat(parts, dim=-1)

        # Ensure correct size
        if obs.shape[1] > self._num_obs:
            obs = obs[:, :self._num_obs]
        elif obs.shape[1] < self._num_obs:
            pad = torch.zeros((obs.shape[0], self._num_obs - obs.shape[1]), device=self._torch_device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """G1-style reward function adapted for Humanoid SNC"""

        # === 1. Velocity tracking rewards (G1 core) ===
        # Linear velocity tracking in xy plane
        lin_vel_error = torch.sqrt(
            (self.vel_loc[:, 0] - self.command_lin_vel_x) ** 2 +
            (self.vel_loc[:, 1] - self.command_lin_vel_y) ** 2
        )
        track_lin_vel_xy = self.cfg.track_lin_vel_xy_weight * torch.exp(-lin_vel_error / self.cfg.velocity_std)

        # Angular velocity tracking
        ang_vel_error = torch.abs(self.angvel_loc[:, 2] - self.command_ang_vel_z)
        track_ang_vel_z = self.cfg.track_ang_vel_z_weight * torch.exp(-ang_vel_error / self.cfg.velocity_std)

        # === 2. Contact and gait rewards ===
        # Update contact info
        self._update_contact_info()

        # Feet air time (encourages proper gait)
        feet_air_time_reward = self.cfg.feet_air_time_weight * torch.mean(
            torch.clamp(self.feet_air_time - self.cfg.feet_air_time_threshold, min=0.0), dim=-1
        )

        # === 3. Stability rewards ===
        # Penalize vertical velocity
        lin_vel_z_penalty = self.cfg.lin_vel_z_weight * (self.velocity[:, 2] ** 2)

        # Penalize roll/pitch angular velocity
        ang_vel_xy_penalty = self.cfg.ang_vel_xy_weight * (
            self.ang_velocity[:, 0] ** 2 + self.ang_velocity[:, 1] ** 2
        )

        # Flat orientation (keep upright)
        flat_orientation_penalty = self.cfg.flat_orientation_weight * (self.roll ** 2 + self.pitch ** 2)

        # === 4. Action penalties (G1-style) ===
        # Action rate (smoothness)
        action_rate = torch.sum((self.actions - self.previous_actions) ** 2, dim=-1)
        action_rate_penalty = self.cfg.action_rate_weight * action_rate

        # Joint acceleration penalty (approximate)
        dof_acc = (self.dof_vel - getattr(self, "_previous_dof_vel", self.dof_vel)) / max(self.cfg.sim.dt * self.cfg.decimation, 1e-6)
        dof_acc_penalty = self.cfg.dof_acc_weight * torch.sum(dof_acc ** 2, dim=-1)

        # Torque penalty
        estimated_torques = self.actions * self.joint_gears.squeeze(0)
        dof_torques_penalty = self.cfg.dof_torques_weight * torch.sum(estimated_torques ** 2, dim=-1)

        # === 5. Joint deviation penalties ===
        # Hip joints (keep them close to default)
        hip_joints = self.dof_pos_scaled[:, [9, 10, 15, 16]]  # hip pitch, roll for both legs
        joint_deviation_hip = self.cfg.joint_deviation_hip_weight * torch.sum(torch.abs(hip_joints), dim=-1)

        # Arm joints
        arm_joints = self.dof_pos_scaled[:, [0, 1, 2, 3, 4, 5, 6, 7]]  # all arm joints
        joint_deviation_arms = self.cfg.joint_deviation_arms_weight * torch.sum(torch.abs(arm_joints), dim=-1)

        # Torso joint
        torso_joint = self.dof_pos_scaled[:, 8]  # pelvis joint
        joint_deviation_torso = self.cfg.joint_deviation_torso_weight * torch.abs(torso_joint)

        # === 6. Joint limits penalty ===
        # Penalize when joints are near limits
        dof_pos_limits_penalty = self.cfg.dof_pos_limits_weight * torch.sum(
            torch.relu(torch.abs(self.dof_pos_scaled) - 0.98), dim=-1
        )

        # === Total reward ===
        total_reward = (
            track_lin_vel_xy +
            track_ang_vel_z +
            feet_air_time_reward +
            lin_vel_z_penalty +
            ang_vel_xy_penalty +
            flat_orientation_penalty +
            action_rate_penalty +
            dof_acc_penalty +
            dof_torques_penalty +
            joint_deviation_hip +
            joint_deviation_arms +
            joint_deviation_torso +
            dof_pos_limits_penalty
        )

        # Termination penalty (G1-style)
        total_reward = torch.where(
            self.reset_terminated,
            torch.ones_like(total_reward) * self.cfg.termination_penalty,
            total_reward
        )

        return total_reward

    def _update_contact_info(self):
        """Update contact forces and feet air time"""
        # Simplified contact detection (in production, use actual contact sensors)
        # For now, assume feet are in contact when robot is stable
        dt = self.cfg.sim.dt * self.cfg.decimation

        # Simple heuristic: feet in contact when vertical velocity is small and height is reasonable
        in_contact = (
            (torch.abs(self.velocity[:, 2]) < 0.5) &  # not jumping/falling fast
            (self.torso_position[:, 2] > self.cfg.termination_height)  # not on ground
        ).unsqueeze(-1).repeat(1, 2)  # both feet

        # Update air time
        self.feet_air_time = torch.where(
            in_contact,
            torch.zeros_like(self.feet_air_time),  # reset when in contact
            self.feet_air_time + dt  # accumulate when in air
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """G1-style termination conditions"""
        self._compute_intermediate_values()

        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Height termination
        height_died = self.torso_position[:, 2] < self.cfg.termination_height

        # Balance termination (more lenient than before)
        balance_died = (
            (torch.abs(self.roll) > self.cfg.max_roll_pitch) |
            (torch.abs(self.pitch) > self.cfg.max_roll_pitch)
        )

        died = height_died | balance_died
        return died, time_out

    def _safe_fix_nan_values(self):
        """Fix NaN values in computed variables"""
        names = [
            "up_proj", "heading_proj", "roll", "pitch", "yaw",
            "angle_to_target", "dof_pos_scaled", "dof_vel",
            "velocity", "ang_velocity", "vel_loc", "angvel_loc",
            "torso_position", "actions",
        ]
        for n in names:
            if hasattr(self, n):
                t = getattr(self, n)
                if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
                    if n in ("roll", "pitch", "yaw", "angle_to_target"):
                        t = torch.nan_to_num(t, nan=0.0, posinf=3.14, neginf=-3.14)
                    elif n in ("up_proj", "heading_proj"):
                        t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
                    elif n in ("velocity", "ang_velocity", "vel_loc", "angvel_loc"):
                        t = torch.nan_to_num(t, nan=0.0, posinf=10.0, neginf=-10.0)
                    elif n in ("dof_pos_scaled",):
                        t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
                    elif n in ("dof_vel",):
                        t = torch.nan_to_num(t, nan=0.0, posinf=10.0, neginf=-10.0)
                    elif n in ("torso_position",):
                        t = torch.nan_to_num(t, nan=0.0, posinf=100.0, neginf=-100.0)
                    elif n in ("actions",):
                        t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
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

    def step(self, actions: torch.Tensor):
        """Enhanced step with G1-style action handling and debug arrows"""
        if actions.dim() == 1:
            actions = actions.view(1, -1)
        if actions.shape[0] == self._num_actions and actions.shape[1] == self.num_envs:
            actions = actions.T
        assert actions.shape[1] == self._num_actions, \
            f"actions shape must be (num_envs, {self._num_actions}), got {tuple(actions.shape)}"

        # Store previous values for rate calculations
        self.previous_actions = self.actions.clone() if hasattr(self, 'actions') else actions.clone()
        self._previous_dof_vel = self.dof_vel.clone() if hasattr(self, 'dof_vel') else None

        obs_dict, rewards, terminated, truncated, info = super().step(actions)
        dones = terminated | truncated

        # Debug arrows (velocity visualization)
        if self._debug_draw is not None:
            try:
                P = self.torso_position.detach().cpu().numpy()
                V = self.velocity.detach().cpu().numpy()
                scale = float(self.cfg.debug_arrow_scale)
                max_lines = int(self.cfg.debug_arrow_max_lines)
                color = tuple(self.cfg.debug_arrow_color)
                self._debug_draw.clear_lines()
                n = min(max_lines, P.shape[0])
                for i in range(n):
                    x0, y0, z0 = P[i].tolist()
                    x1, y1, z1 = (P[i] + V[i] * scale).tolist()
                    self._debug_draw.draw_line([x0, y0, z0], [x1, y1, z1], color)
            except Exception:
                pass

        return obs_dict["policy"], rewards, dones, info