# SPDX-License-Identifier: BSD-3-Clause
"""
Simple and SAFE Humanoid SNC Environment
Focus: No NaN, no explosions, just basic walking
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
class HumanoidSNCSimpleEnvCfg(DirectRLEnvCfg):
    """SIMPLE and SAFE configuration"""

    # === Timing ===
    episode_length_s = 10.0
    decimation = 2
    action_scale = 0.5  # Conservative

    # === Spaces ===
    action_space = 21
    observation_space = 75  # Keep it simple

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

    # Joint gears - conservative
    joint_gears: list[float] = [
        30.0, 30.0, 30.0, 20.0,  # arms - gentle
        30.0, 30.0, 30.0, 20.0,
        30.0,  # torso
        40.0, 80.0, 40.0,  # left leg - moderate
        60.0, 20.0, 20.0,
        40.0, 80.0, 40.0,  # right leg
        60.0, 20.0, 20.0,
    ]

    # === SIMPLE REWARDS (NO COMPLEX MATH) ===
    alive_reward: float = 1.0           # Simple alive bonus
    forward_reward: float = 2.0         # Reward for moving forward
    up_reward: float = 1.0              # Reward for staying upright
    action_penalty: float = 0.01        # Small action penalty
    fall_penalty: float = 10.0          # Penalty for falling

    # === Target velocity ===
    target_velocity: float = 0.5  # m/s forward

    # === Termination ===
    termination_height: float = 0.6
    max_roll_pitch: float = 1.2  # ~70 degrees

    # === Scaling ===
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1

    # === Debug ===
    enable_debug_arrows: bool = True
    debug_arrow_scale: float = 0.3

class HumanoidSNCSimpleEnv(LocomotionEnv):
    """SIMPLE Humanoid SNC Environment - No complex math, no NaN"""

    cfg: HumanoidSNCSimpleEnvCfg

    def __init__(self, cfg: HumanoidSNCSimpleEnvCfg, render_mode: str | None = None, **kwargs):
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

        # Simple tracking
        self.previous_actions = torch.zeros((self.num_envs, self._num_actions), device=self._torch_device)

        # Debug arrows
        self._debug_draw = None
        if self.cfg.enable_debug_arrows:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                print("🎯 Simple debug arrows enabled!")
            except Exception as e:
                print(f"[DEBUG-ARROWS] unavailable: {e}")

        print("🚀 SIMPLE Humanoid SNC Environment initialized!")

    def _get_observations(self) -> dict:
        """SIMPLE observations - no complex calculations"""
        try:
            self._compute_intermediate_values()
        except Exception as e:
            print(f"[_get_observations] error: {e}")

        # Fix any NaN/inf immediately
        self._safe_fix_everything()

        # Keep it simple
        parts = [
            # Basic info
            self.torso_position[:, 2:3],  # height
            self.velocity,                # world velocity (3D)
            self.ang_velocity * self.cfg.angular_velocity_scale,  # angular velocity (3D)

            # Orientation (simple)
            torch.sin(self.roll).unsqueeze(-1),
            torch.cos(self.roll).unsqueeze(-1),
            torch.sin(self.pitch).unsqueeze(-1),
            torch.cos(self.pitch).unsqueeze(-1),
            torch.sin(self.yaw).unsqueeze(-1),
            torch.cos(self.yaw).unsqueeze(-1),

            # Joint states
            self.dof_pos_scaled,
            self.dof_vel * self.cfg.dof_vel_scale,

            # Actions
            self.actions,
        ]

        obs = torch.cat(parts, dim=-1)

        # Ensure correct size
        if obs.shape[1] > self._num_obs:
            obs = obs[:, :self._num_obs]
        elif obs.shape[1] < self._num_obs:
            pad = torch.zeros((obs.shape[0], self._num_obs - obs.shape[1]), device=self._torch_device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

        # Final safety
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = torch.clamp(obs, min=-10.0, max=10.0)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """SIMPLE reward function - no complex math, no explosions"""

        # Fix everything first
        self._safe_fix_everything()

        # === 1. Alive reward (simple) ===
        alive = torch.ones(self.num_envs, device=self._torch_device) * self.cfg.alive_reward

        # === 2. Forward movement reward (simple) ===
        forward_speed = self.velocity[:, 0]  # x-direction speed
        # Simple linear reward (no exponentials!)
        forward_reward = self.cfg.forward_reward * torch.clamp(forward_speed / self.cfg.target_velocity, -1.0, 2.0)

        # === 3. Up reward (simple) ===
        # Simple height-based reward
        height_good = torch.clamp((self.torso_position[:, 2] - 0.5) / 0.5, 0.0, 1.0)
        up_reward = self.cfg.up_reward * height_good

        # === 4. Action penalty (simple) ===
        action_penalty = -self.cfg.action_penalty * torch.mean(self.actions ** 2, dim=-1)

        # === 5. Total reward (SIMPLE sum) ===
        total_reward = alive + forward_reward + up_reward + action_penalty

        # === SAFETY CLAMPS ===
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=10.0, neginf=-10.0)
        total_reward = torch.clamp(total_reward, min=-20.0, max=20.0)

        # Fall penalty (only when terminated)
        total_reward = torch.where(
            self.reset_terminated,
            torch.ones_like(total_reward) * (-self.cfg.fall_penalty),
            total_reward
        )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """SIMPLE termination conditions"""
        self._compute_intermediate_values()
        self._safe_fix_everything()

        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Simple termination conditions
        height_died = self.torso_position[:, 2] < self.cfg.termination_height

        # Simple roll/pitch check (no complex angle math)
        roll_died = torch.abs(self.roll) > self.cfg.max_roll_pitch
        pitch_died = torch.abs(self.pitch) > self.cfg.max_roll_pitch

        died = height_died | roll_died | pitch_died
        return died, time_out

    def _safe_fix_everything(self):
        """Fix ALL potential NaN/inf issues"""
        # List of all variables that might have NaN
        vars_to_fix = [
            "torso_position", "velocity", "ang_velocity",
            "roll", "pitch", "yaw",
            "dof_pos_scaled", "dof_vel", "actions",
            "up_proj", "heading_proj"
        ]

        for var_name in vars_to_fix:
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                if isinstance(var, torch.Tensor):
                    # Fix NaN/inf
                    var = torch.nan_to_num(var, nan=0.0, posinf=1e3, neginf=-1e3)
                    # Clamp to reasonable ranges
                    if var_name in ["torso_position"]:
                        var = torch.clamp(var, min=-100.0, max=100.0)
                    elif var_name in ["velocity", "ang_velocity"]:
                        var = torch.clamp(var, min=-50.0, max=50.0)
                    elif var_name in ["roll", "pitch", "yaw"]:
                        var = torch.clamp(var, min=-3.15, max=3.15)
                    elif var_name in ["dof_pos_scaled"]:
                        var = torch.clamp(var, min=-2.0, max=2.0)
                    elif var_name in ["dof_vel"]:
                        var = torch.clamp(var, min=-100.0, max=100.0)
                    elif var_name in ["actions"]:
                        var = torch.clamp(var, min=-1.0, max=1.0)
                    else:
                        var = torch.clamp(var, min=-10.0, max=10.0)

                    setattr(self, var_name, var)

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
        """SIMPLE step with safety"""
        # Fix action shape
        if actions.dim() == 1:
            actions = actions.view(1, -1)
        if actions.shape[0] == self._num_actions and actions.shape[1] == self.num_envs:
            actions = actions.T

        # Safety clamp actions immediately
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)

        # Store previous
        self.previous_actions = self.actions.clone() if hasattr(self, 'actions') else actions.clone()

        obs_dict, rewards, terminated, truncated, info = super().step(actions)
        dones = terminated | truncated

        # Debug arrows
        if self._debug_draw is not None:
            try:
                P = self.torso_position.detach().cpu().numpy()
                V = self.velocity.detach().cpu().numpy()
                scale = float(self.cfg.debug_arrow_scale)
                self._debug_draw.clear_lines()
                n = min(32, P.shape[0])  # Limit for performance
                for i in range(n):
                    x0, y0, z0 = P[i].tolist()
                    x1, y1, z1 = (P[i] + V[i] * scale).tolist()
                    self._debug_draw.draw_line([x0, y0, z0], [x1, y1, z1], (0.0, 1.0, 0.0, 1.0))
            except Exception:
                pass

        return obs_dict["policy"], rewards, dones, info