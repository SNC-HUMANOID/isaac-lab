# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

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


@configclass
class HumanoidSNCEnvCfg(DirectRLEnvCfg):
    # timing
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0

    # ใช้ตัวเลขแทนการประกาศ gym spaces ตรง ๆ
    num_actions = 21
    num_observations = 75

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
    robot: ArticulationCfg = HUMANOID_SNC_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # เกียร์ตามจำนวนข้อต่อ
    joint_gears: list[float] = [
        67.5, 67.5, 67.5, 45.0,
        67.5, 67.5, 67.5, 45.0,
        67.5,
        45.0, 135.0, 45.0,
        90.0, 22.5, 22.5,
        45.0, 135.0, 45.0,
        90.0, 22.5, 22.5,
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


class HumanoidSNCEnv(LocomotionEnv):
    """Environment ที่ยึด LocomotionEnv แต่บังคับ shape ให้ถูกต้องสำหรับ RSL-RL."""

    cfg: HumanoidSNCEnvCfg

    def __init__(self, cfg: HumanoidSNCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.device = self.sim.device if hasattr(self, "sim") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_actions = int(self.cfg.num_actions)
        self._num_obs = int(self.cfg.num_observations)

        # -------------------- Gym spaces --------------------
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._num_obs,), dtype=np.float32
        )

        # สำหรับ runner
        self.num_actions = self._num_actions

        # -------------------- joint_gears -> (1, num_actions) --------------------
        assert len(self.cfg.joint_gears) == self._num_actions, \
            f"joint_gears ({len(self.cfg.joint_gears)}) != num_actions ({self._num_actions})"

        self.joint_gears = torch.tensor(
            self.cfg.joint_gears, device=self.device, dtype=torch.float32
        ).view(1, -1)  # (1, num_actions)

    # ----------------------------------------------------------------------------------
    # Safety helpers
    # ----------------------------------------------------------------------------------
    def _safe_fix_nan_values(self):
        # ฟังก์ชันเล็ก ๆ ล้าง NaN/Inf ในตัวแปรสำคัญ
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
                    # กำหนดคลิปคร่าว ๆ ให้ไปต่อได้
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

    # ----------------------------------------------------------------------------------
    # RSL-RL compatibility
    # ----------------------------------------------------------------------------------
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_dict = self._get_observations()
        obs_tensor = obs_dict["policy"]
        extras = {
            "observations": obs_dict,
            "episode_length_buf": getattr(self, "episode_length_buf", torch.zeros(self.num_envs, device=self.device)),
        }
        return obs_tensor, extras

    def step(self, actions: torch.Tensor):
        """
        บังคับ shape actions เป็น (num_envs, num_actions) เสมอ
        และปล่อยให้ parent จัดการ apply_action + physics
        """
        if actions.dim() == 1:
            actions = actions.view(1, -1)
        # เคส (num_actions, num_envs) -> transpose
        if actions.shape[0] == self._num_actions and actions.shape[1] == self.num_envs:
            actions = actions.T
        # เช็คขั้นสุดท้าย
        assert actions.shape[1] == self._num_actions, \
            f"actions shape must be (num_envs, {self._num_actions}), got {tuple(actions.shape)}"

        obs_dict, rewards, terminated, truncated, info = super().step(actions)
        dones = terminated | truncated
        obs_tensor = obs_dict["policy"]
        return obs_tensor, rewards, dones, info

    # ----------------------------------------------------------------------------------
    # Observations / Rewards
    # ----------------------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """รวม observation ให้ได้มิติเท่ากับ cfg.num_observations (75)"""
        # ให้ parent คำนวณค่ากลาง ๆ ก่อน (ตำแหน่ง/ความเร็ว/ฯลฯ)
        try:
            self._compute_intermediate_values()
        except Exception as e:
            print(f"[_get_observations] compute_intermediate_values error: {e}")

        self._safe_fix_nan_values()

        from isaaclab_tasks.direct.locomotion.locomotion_env import normalize_angle

        # ประกอบ observation (ต้องมี actions ติดท้ายเพื่อช่วย stabilize policy ในช่วงแรก)
        obs_parts = [
            self.torso_position[:, 2:3],                       # height (1)
            self.vel_loc,                                      # (3)
            self.angvel_loc * self.cfg.angular_velocity_scale, # (3)
            normalize_angle(self.yaw).unsqueeze(-1),           # (1)
            normalize_angle(self.roll).unsqueeze(-1),          # (1)
            normalize_angle(self.angle_to_target).unsqueeze(-1),# (1)
            self.up_proj.unsqueeze(-1),                        # (1)
            self.heading_proj.unsqueeze(-1),                   # (1)
            self.dof_pos_scaled,                               # (num_dof)
            self.dof_vel * self.cfg.dof_vel_scale,             # (num_dof)
            self.actions,                                      # (num_actions)
        ]
        obs = torch.cat(obs_parts, dim=-1)

        # ตัด/เติม ให้ได้พอดี 75 (กันกรณีจำนวน DOF ต่างไปเล็กน้อย)
        target = self._num_obs
        if obs.shape[1] > target:
            obs = obs[:, :target]
        elif obs.shape[1] < target:
            pad = torch.zeros((obs.shape[0], target - obs.shape[1]), device=self.device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return {"policy": obs}
