# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import os, csv, math
import gymnasium as gym
import numpy as np
import torch

from isaaclab_assets.robots.humanoid_snc import HUMANOID_SNC_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False

@configclass
class HumanoidSNCEnvCfg(DirectRLEnvCfg):
    # --- timing / controls ---
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0

    # ต้องมีสองฟิลด์นี้เพื่อผ่าน cfg.validate()
    action_space = 21
    observation_space = 75  # ถ้า add_imu_to_observation=True → เพิ่ม +6 เป็น 81

    # --- simulation / scene ---
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

    # --- dynamics / rewards ---
    joint_gears: list[float] = [
        67.5, 67.5, 67.5, 45.0,
        67.5, 67.5, 67.5, 45.0,
        67.5,
        45.0, 135.0, 45.0,
        90.0, 22.5, 22.5,
        45.0, 135.0, 45.0,
        90.0, 22.5, 22.5,
    ]
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

    # --- Debug arrows options ---
    enable_debug_arrows: bool = True
    debug_arrow_scale: float = 0.3
    debug_arrow_max_lines: int = 256
    debug_arrow_color: tuple[float, float, float, float] = (0.1, 0.7, 1.0, 1.0)  # RGBA

    # --- Virtual IMU options ---
    enable_virtual_imu: bool = True
    add_imu_to_observation: bool = False
    gravity: float = 9.81

    # --- Logging options ---
    enable_episode_logging: bool = True
    log_dir: str = "logs"
    kpi_csv: str = "humanoid_snc_kpis.csv"
    tb_dir: str = "tb"  # tensorboard subdir under log_dir

class HumanoidSNCEnv(LocomotionEnv):
    """LocomotionEnv + debug arrows + virtual IMU + episode KPIs logging (CSV/TensorBoard)."""

    cfg: HumanoidSNCEnvCfg

    def __init__(self, cfg: HumanoidSNCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # device ภายในสำหรับสร้างเทนเซอร์
        sim_dev = self.sim.device if hasattr(self, "sim") else ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = torch.device(sim_dev) if isinstance(sim_dev, str) else sim_dev

        self._num_actions = int(self.cfg.action_space)
        self._num_obs     = int(self.cfg.observation_space)

        # Gym spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._num_obs,), dtype=np.float32)
        self.num_actions = self._num_actions

        # joint_gears -> (1, num_actions)
        assert len(self.cfg.joint_gears) == self._num_actions, \
            f"joint_gears ({len(self.cfg.joint_gears)}) != num_actions ({self._num_actions})"
        self.joint_gears = torch.tensor(self.cfg.joint_gears, device=self._torch_device, dtype=torch.float32).view(1, -1)

        # Debug draw
        self._debug_draw = None
        if self.cfg.enable_debug_arrows:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as e:
                print(f"[DEBUG-ARROWS] unavailable: {e}")

        # IMU buffers
        self._prev_vel_world = torch.zeros((self.num_envs, 3), device=self._torch_device)
        self._prev_vel_loc   = torch.zeros((self.num_envs, 3), device=self._torch_device)

        if self.cfg.add_imu_to_observation:
            need = 6
            if self._num_obs < 75 + need:
                print(f"[IMU] WARNING: add_imu_to_observation=True แต่ observation_space={self._num_obs} < 81 (75+6). "
                      f"จะถูก pad/clip อัตโนมัติ แต่ควรปรับ cfg.observation_space ให้พอดี.")

        # ---------- Episode KPI logging ----------
        self._log_enabled = bool(self.cfg.enable_episode_logging)
        self._log_dir = self.cfg.log_dir
        self._csv_path = os.path.join(self._log_dir, self.cfg.kpi_csv)
        self._tb = None
        os.makedirs(self._log_dir, exist_ok=True)
        if self._log_enabled:
            # เตรียม CSV header
            if not os.path.exists(self._csv_path):
                with open(self._csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "global_episode", "env_id", "episode_idx",
                        "return", "ep_len_steps", "ep_len_s", "success", "fall_reason",
                        "up_proj_mean", "height_mean",
                        "roll_rms_deg", "pitch_rms_deg",
                        "speed_xy_mean", "action_l2_mean", "elec_cost_mean",
                    ])
            # TensorBoard
            if _HAS_TB:
                self._tb = SummaryWriter(log_dir=os.path.join(self._log_dir, self.cfg.tb_dir))
            else:
                print("[LOG] TensorBoard not available; CSV only.")

        # ต่อ-env episode accumulators
        N = self.num_envs
        self._ep_idx         = torch.zeros(N, dtype=torch.long, device=self._torch_device)  # counter per env
        self._ep_steps       = torch.zeros(N, dtype=torch.long, device=self._torch_device)
        self._ep_return      = torch.zeros(N, dtype=torch.float32, device=self._torch_device)

        self._acc_up_proj    = torch.zeros(N, dtype=torch.float32, device=self._torch_device)
        self._acc_height     = torch.zeros(N, dtype=torch.float32, device=self._torch_device)
        self._acc_roll2      = torch.zeros(N, dtype=torch.float32, device=self._torch_device)
        self._acc_pitch2     = torch.zeros(N, dtype=torch.float32, device=self._torch_device)
        self._acc_speed_xy   = torch.zeros(N, dtype=torch.float32, device=self._torch_device)
        self._acc_action_l2  = torch.zeros(N, dtype=torch.float32, device=self._torch_device)
        self._acc_elec_cost  = torch.zeros(N, dtype=torch.float32, device=self._torch_device)

        self._global_episode = 0  # นับจำนวนเอพิโสดทั้งหมดที่บันทึก

    # -------------- Safety helpers --------------
    def _safe_fix_nan_values(self):
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

    # -------------- RSL-RL compatibility --------------
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict = self._get_observations()
        obs_tensor = obs_dict["policy"]
        extras = {
            "observations": obs_dict,
            "episode_length_buf": getattr(self, "episode_length_buf", torch.zeros(self.num_envs, device=self._torch_device)),
        }
        return obs_tensor, extras

    def step(self, actions: torch.Tensor):
        """บังคับ actions shape, ทำ IMU, วาดลูกศร, สะสม KPI และเขียน log เมื่อจบเอพิโสด"""
        if actions.dim() == 1:
            actions = actions.view(1, -1)
        if actions.shape[0] == self._num_actions and actions.shape[1] == self.num_envs:
            actions = actions.T
        assert actions.shape[1] == self._num_actions, \
            f"actions shape must be (num_envs, {self._num_actions}), got {tuple(actions.shape)}"

        obs_dict, rewards, terminated, truncated, info = super().step(actions)
        dones = terminated | truncated

        # --------- Virtual IMU (info only) ---------
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

        # --------- Debug arrows (velocity) ---------
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

        # --------- KPIs accumulation & logging ---------
        try:
            self._accumulate_kpis(rewards)
            self._flush_done_episodes_to_logs(terminated, truncated)
        except Exception:
            pass  # never break training for logging

        return obs_dict["policy"], rewards, dones, info

    # -------------- Observations --------------
    def _get_observations(self) -> dict:
        try:
            self._compute_intermediate_values()
        except Exception as e:
            print(f"[_get_observations] compute_intermediate_values error: {e}")

        self._safe_fix_nan_values()
        from isaaclab_tasks.direct.locomotion.locomotion_env import normalize_angle

        parts = [
            self.torso_position[:, 2:3],
            self.vel_loc,
            self.angvel_loc * self.cfg.angular_velocity_scale,
            normalize_angle(self.yaw).unsqueeze(-1),
            normalize_angle(self.roll).unsqueeze(-1),
            normalize_angle(self.angle_to_target).unsqueeze(-1),
            self.up_proj.unsqueeze(-1),
            self.heading_proj.unsqueeze(-1),
            self.dof_pos_scaled,
            self.dof_vel * self.cfg.dof_vel_scale,
            self.actions,
        ]

        if self.cfg.enable_virtual_imu and self.cfg.add_imu_to_observation:
            dt = float(self.cfg.sim.dt * self.cfg.decimation)
            if hasattr(self, "vel_loc"):
                a_body = (self.vel_loc - self._prev_vel_loc) / max(dt, 1e-6)
            else:
                a_body = torch.zeros((self.num_envs, 3), device=self._torch_device)
            gyro_body = self.angvel_loc if hasattr(self, "angvel_loc") else torch.zeros_like(a_body)
            parts.extend([gyro_body, a_body])

        obs = torch.cat(parts, dim=-1)

        if obs.shape[1] > self._num_obs:
            obs = obs[:, :self._num_obs]
        elif obs.shape[1] < self._num_obs:
            pad = torch.zeros((obs.shape[0], self._num_obs - obs.shape[1]), device=self._torch_device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return {"policy": obs}

    # -------------- Private: KPIs --------------
    def _accumulate_kpis(self, rewards: torch.Tensor):
        """สะสมสถิติรายสเต็ปสำหรับแต่ละ env"""
        N = self.num_envs
        dt = float(self.cfg.sim.dt * self.cfg.decimation)

        # safety
        for name in ("up_proj", "torso_position", "roll", "pitch", "velocity", "actions", "dof_vel"):
            if not hasattr(self, name):
                return

        height = self.torso_position[:, 2]
        roll = self.roll
        pitch = self.pitch
        vel = self.velocity
        actions = self.actions
        dof_vel = self.dof_vel

        speed_xy = torch.linalg.norm(vel[:, :2], dim=-1)  # m/s in XY
        action_l2 = torch.mean(actions ** 2, dim=-1)      # mean over joints
        elec = torch.mean(torch.abs(actions * dof_vel * 0.1), dim=-1)  # proxy electricity

        # accumulate
        self._ep_steps += 1
        self._ep_return += rewards
        self._acc_up_proj += self.up_proj
        self._acc_height += height
        self._acc_roll2 += roll ** 2
        self._acc_pitch2 += pitch ** 2
        self._acc_speed_xy += speed_xy
        self._acc_action_l2 += action_l2
        self._acc_elec_cost += elec

    def _flush_done_episodes_to_logs(self, terminated: torch.Tensor, truncated: torch.Tensor):
        """เมื่อเอพิโสดใดจบ (terminated หรือ truncated) → เขียน CSV/TB และรีเซ็ตตัวสะสม"""
        if not self._log_enabled:
            # reset counters on done anyway
            done_mask = (terminated | truncated)
            if done_mask.any():
                self._reset_episode_accumulators(done_mask)
            return

        done_mask = (terminated | truncated)
        if not done_mask.any():
            return

        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        idxs = torch.nonzero(done_mask, as_tuple=False).flatten()
        success_mask = truncated[done_mask] & (~terminated[done_mask])  # time-limit = success (โดยนิยามนี้)

        rows = []
        for k, env_id in enumerate(idxs.tolist()):
            steps = int(self._ep_steps[env_id].item())
            ep_time = steps * dt
            ret = float(self._ep_return[env_id].item())
            success = bool(success_mask[k].item())

            # means
            denom = max(steps, 1)
            up_mean = float(self._acc_up_proj[env_id].item() / denom)
            h_mean = float(self._acc_height[env_id].item() / denom)
            roll_rms = math.degrees(math.sqrt(float(self._acc_roll2[env_id].item() / denom)))
            pitch_rms = math.degrees(math.sqrt(float(self._acc_pitch2[env_id].item() / denom)))
            spd_mean = float(self._acc_speed_xy[env_id].item() / denom)
            act_mean = float(self._acc_action_l2[env_id].item() / denom)
            elec_mean = float(self._acc_elec_cost[env_id].item() / denom)

            # fall reason (คร่าว ๆ) — ต่ำกว่า height
            fall_reason = ""
            if not success and terminated[env_id]:
                if hasattr(self, "torso_position") and float(self.torso_position[env_id, 2].item()) < self.cfg.termination_height:
                    fall_reason = "height"
                else:
                    fall_reason = "terminate"

            # global episode id
            self._global_episode += 1
            ep_idx = int(self._ep_idx[env_id].item())

            rows.append([
                self._global_episode, env_id, ep_idx,
                ret, steps, ep_time, int(success), fall_reason,
                up_mean, h_mean, roll_rms, pitch_rms, spd_mean, act_mean, elec_mean
            ])

            # TensorBoard (ต่อ episode)
            if self._tb is not None:
                g = self._global_episode
                self._tb.add_scalar("episode/return", ret, g)
                self._tb.add_scalar("episode/len_steps", steps, g)
                self._tb.add_scalar("episode/len_seconds", ep_time, g)
                self._tb.add_scalar("episode/success", int(success), g)
                self._tb.add_scalar("kpi/up_proj_mean", up_mean, g)
                self._tb.add_scalar("kpi/height_mean", h_mean, g)
                self._tb.add_scalar("kpi/roll_rms_deg", roll_rms, g)
                self._tb.add_scalar("kpi/pitch_rms_deg", pitch_rms, g)
                self._tb.add_scalar("kpi/speed_xy_mean", spd_mean, g)
                self._tb.add_scalar("kpi/action_l2_mean", act_mean, g)
                self._tb.add_scalar("kpi/elec_cost_mean", elec_mean, g)

        # write CSV batch
        if rows:
            with open(self._csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerows(rows)

        # reset accumulators for those envs & increase per-env episode idx
        self._reset_episode_accumulators(done_mask)
        self._ep_idx[done_mask] += 1

    def _reset_episode_accumulators(self, mask: torch.Tensor):
        """รีเซ็ตตัวสะสมสำหรับ env ที่จบเอพิโสดแล้ว"""
        self._ep_steps[mask] = 0
        self._ep_return[mask] = 0.0
        self._acc_up_proj[mask] = 0.0
        self._acc_height[mask] = 0.0
        self._acc_roll2[mask] = 0.0
        self._acc_pitch2[mask] = 0.0
        self._acc_speed_xy[mask] = 0.0
        self._acc_action_l2[mask] = 0.0
        self._acc_elec_cost[mask] = 0.0
