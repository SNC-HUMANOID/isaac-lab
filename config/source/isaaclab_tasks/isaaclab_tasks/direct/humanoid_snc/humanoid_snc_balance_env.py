# SPDX-License-Identifier: BSD-3-Clause
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
class HumanoidSNCBalanceEnvCfg(DirectRLEnvCfg):
    """Improved configuration for balance and walking training"""

    # === Timing ===
    episode_length_s = 5.0  # เริ่มสั้นๆ เพื่อเรียนรู้เร็ว
    decimation = 2
    action_scale = 0.5  # ลดลงเพื่อความนุ่มนวล

    # === Action/Observation Space ===
    action_space = 21
    observation_space = 99  # เพิ่มขึ้นเพื่อ contact forces, balance info และ IMU (6 channels)

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
        50.0, 50.0, 50.0, 35.0,  # ลดลงเล็กน้อย
        50.0, 50.0, 50.0, 35.0,
        50.0,
        35.0, 100.0, 35.0,
        70.0, 20.0, 20.0,
        35.0, 100.0, 35.0,
        70.0, 20.0, 20.0,
    ]

    # === Reward Weights (ปรับให้เน้น balance) ===
    up_weight: float = 1.0  # เพิ่มขึ้น
    balance_weight: float = 2.0  # ใหม่ - สำหรับ roll/pitch
    heading_weight: float = 0.3  # ลดลง - ไม่เน้นการเดินในตอนแรก
    energy_cost_scale: float = 0.02  # ลดลง
    actions_cost_scale: float = 0.005  # ลดลง
    alive_reward_scale: float = 3.0  # เพิ่มขึ้น
    contact_reward_scale: float = 0.5  # ใหม่ - สำหรับเท้าสัมผัสพื้น

    # === Velocity Scaling ===
    dof_vel_scale: float = 0.05  # ลดลง
    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

    # === Termination ===
    termination_height: float = 0.65  # ลดลงจาก 0.8
    death_cost: float = -5.0  # ลดลง

    # === Balance Thresholds ===
    max_roll_pitch: float = 0.7  # radians (~40 degrees)
    target_height: float = 1.0  # target standing height

    # === Contact Settings ===
    contact_bodies: list[str] = ["left_foot", "right_foot"]  # ชื่อของเท้า

    # === Debug arrows options ===
    enable_debug_arrows: bool = True
    debug_arrow_scale: float = 0.5
    debug_arrow_max_lines: int = 256
    debug_arrow_color: tuple[float, float, float, float] = (0.1, 0.7, 1.0, 1.0)  # RGBA

    # === Virtual IMU options ===
    enable_virtual_imu: bool = True
    add_imu_to_observation: bool = True  # เปิดเพื่อเห็น IMU data
    gravity: float = 9.81

class HumanoidSNCBalanceEnv(LocomotionEnv):
    """Enhanced environment for balance and walking with improved rewards"""

    cfg: HumanoidSNCBalanceEnvCfg

    def __init__(self, cfg: HumanoidSNCBalanceEnvCfg, render_mode: str | None = None, **kwargs):
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

        # Contact force tracking
        self._contact_forces = torch.zeros((self.num_envs, 2, 3), device=self._torch_device)  # left, right foot

        # Balance tracking
        self._prev_height = torch.zeros(self.num_envs, device=self._torch_device)

        # Debug draw setup
        self._debug_draw = None
        if self.cfg.enable_debug_arrows:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                print("🎯 Debug arrows enabled!")
            except Exception as e:
                print(f"[DEBUG-ARROWS] unavailable: {e}")

        # IMU buffers for virtual IMU
        self._prev_vel_world = torch.zeros((self.num_envs, 3), device=self._torch_device)
        self._prev_vel_loc = torch.zeros((self.num_envs, 3), device=self._torch_device)

        if self.cfg.add_imu_to_observation:
            print("📱 Virtual IMU enabled and added to observations!")

    def _get_observations(self) -> dict:
        """Enhanced observations including contact forces and balance info"""
        try:
            self._compute_intermediate_values()
        except Exception as e:
            print(f"[_get_observations] compute_intermediate_values error: {e}")

        self._safe_fix_nan_values()

        # Basic observations
        parts = [
            self.torso_position[:, 2:3],  # height
            self.vel_loc,  # local velocity
            self.angvel_loc * self.cfg.angular_velocity_scale,  # local angular velocity
            normalize_angle(self.yaw).unsqueeze(-1),
            normalize_angle(self.roll).unsqueeze(-1),
            normalize_angle(self.pitch).unsqueeze(-1),  # เพิ่ม pitch explicit
            normalize_angle(self.angle_to_target).unsqueeze(-1),
            self.up_proj.unsqueeze(-1),
            self.heading_proj.unsqueeze(-1),
            self.dof_pos_scaled,
            self.dof_vel * self.cfg.dof_vel_scale,
            self.actions,
        ]

        # Enhanced balance info
        height_diff = (self.torso_position[:, 2] - self.cfg.target_height).unsqueeze(-1)
        parts.append(height_diff)

        # Contact forces (simplified)
        contact_info = self._get_contact_info()
        parts.append(contact_info)

        # Center of mass deviation (approximate)
        com_deviation = torch.stack([self.roll, self.pitch], dim=-1)
        parts.append(com_deviation)

        # IMU data (if enabled)
        if self.cfg.enable_virtual_imu and self.cfg.add_imu_to_observation:
            dt = float(self.cfg.sim.dt * self.cfg.decimation)
            if hasattr(self, "vel_loc"):
                a_body = (self.vel_loc - self._prev_vel_loc) / max(dt, 1e-6)
            else:
                a_body = torch.zeros((self.num_envs, 3), device=self._torch_device)
            gyro_body = self.angvel_loc if hasattr(self, "angvel_loc") else torch.zeros_like(a_body)
            parts.extend([gyro_body, a_body])

        obs = torch.cat(parts, dim=-1)

        # Ensure correct size
        if obs.shape[1] > self._num_obs:
            obs = obs[:, :self._num_obs]
        elif obs.shape[1] < self._num_obs:
            pad = torch.zeros((obs.shape[0], self._num_obs - obs.shape[1]), device=self._torch_device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return {"policy": obs}

    def _get_contact_info(self) -> torch.Tensor:
        """Get simplified contact information"""
        # Simplified: use foot height as proxy for contact
        try:
            # Get foot positions (assume they are tracked in robot data)
            # For now, use simplified approximation
            left_contact = (self.torso_position[:, 2] < 1.2).float()  # rough estimate
            right_contact = (self.torso_position[:, 2] < 1.2).float()
            return torch.stack([left_contact, right_contact], dim=-1)
        except:
            # Fallback
            return torch.zeros((self.num_envs, 2), device=self._torch_device)

    def _get_rewards(self) -> torch.Tensor:
        """Enhanced reward function focused on balance and stability"""

        # === 1. Alive reward ===
        alive_reward = torch.ones(self.num_envs, device=self._torch_device) * self.cfg.alive_reward_scale

        # === 2. Height reward (standing upright) ===
        height_error = torch.abs(self.torso_position[:, 2] - self.cfg.target_height)
        height_reward = torch.exp(-3.0 * height_error)

        # === 3. Up projection reward (more lenient) ===
        up_reward = torch.where(
            self.up_proj > 0.85,
            self.cfg.up_weight,
            self.cfg.up_weight * (self.up_proj / 0.85)
        )

        # === 4. Balance reward (roll/pitch) ===
        balance_error = torch.sqrt(self.roll**2 + self.pitch**2)
        balance_reward = self.cfg.balance_weight * torch.exp(-5.0 * balance_error)

        # === 5. Velocity penalty (encourage staying still initially) ===
        velocity_penalty = -0.1 * torch.norm(self.velocity, dim=-1)

        # === 6. Action smoothness ===
        action_penalty = -self.cfg.actions_cost_scale * torch.sum(self.actions**2, dim=-1)

        # === 7. Energy penalty ===
        energy_penalty = -self.cfg.energy_cost_scale * torch.sum(
            torch.abs(self.actions * self.dof_vel * self.cfg.dof_vel_scale), dim=-1
        )

        # === 8. Contact reward (both feet on ground) ===
        contact_info = self._get_contact_info()
        contact_reward = self.cfg.contact_reward_scale * torch.sum(contact_info, dim=-1)

        # === 9. Small heading reward (only if balanced) ===
        balanced_mask = (balance_error < 0.3).float()
        heading_reward = self.cfg.heading_weight * self.heading_proj * balanced_mask

        # === Total reward ===
        total_reward = (
            alive_reward +
            height_reward +
            up_reward +
            balance_reward +
            velocity_penalty +
            action_penalty +
            energy_penalty +
            contact_reward +
            heading_reward
        )

        # Death penalty
        total_reward = torch.where(
            self.reset_terminated,
            torch.ones_like(total_reward) * self.cfg.death_cost,
            total_reward
        )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Enhanced termination conditions"""
        self._compute_intermediate_values()

        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Height termination
        height_died = self.torso_position[:, 2] < self.cfg.termination_height

        # Balance termination (extreme angles)
        balance_died = (
            (torch.abs(self.roll) > self.cfg.max_roll_pitch) |
            (torch.abs(self.pitch) > self.cfg.max_roll_pitch)
        )

        # Combine death conditions
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
        """Enhanced step with improved action handling + debug arrows + IMU"""
        if actions.dim() == 1:
            actions = actions.view(1, -1)
        if actions.shape[0] == self._num_actions and actions.shape[1] == self.num_envs:
            actions = actions.T
        assert actions.shape[1] == self._num_actions, \
            f"actions shape must be (num_envs, {self._num_actions}), got {tuple(actions.shape)}"

        obs_dict, rewards, terminated, truncated, info = super().step(actions)
        dones = terminated | truncated

        # --------- Virtual IMU (add to info) ---------
        if self.cfg.enable_virtual_imu:
            try:
                dt = float(self.cfg.sim.dt * self.cfg.decimation)
                v_world = self.velocity  # (N,3)
                dv_world = (v_world - self._prev_vel_world) / max(dt, 1e-6)
                g = torch.tensor([0.0, 0.0, -self.cfg.gravity], device=self._torch_device).view(1, 3)
                a_world = dv_world - g

                if hasattr(self, "vel_loc"):
                    a_body = (self.vel_loc - self._prev_vel_loc) / max(dt, 1e-6)
                else:
                    a_body = a_world

                gyro_body = self.angvel_loc if hasattr(self, "angvel_loc") else self.ang_velocity

                info["imu"] = {"gyro": gyro_body.detach(), "accel": a_body.detach()}
                self._prev_vel_world = v_world.detach()
                if hasattr(self, "vel_loc"):
                    self._prev_vel_loc = self.vel_loc.detach()
            except Exception:
                pass

        # --------- Debug arrows (velocity visualization) ---------
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