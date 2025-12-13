# SPDX-License-Identifier: BSD-3-Clause
"""
🚶 FORCE WALK Humanoid SNC Environment
This environment FORCES the robot to walk by giving huge penalties for not moving
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
class HumanoidSNCForceWalkEnvCfg(DirectRLEnvCfg):
    """Configuration that FORCES walking"""

    # === Timing ===
    episode_length_s = 8.0  # Shorter episodes to learn faster
    decimation = 2
    action_scale = 1.0  # Full action range

    # === Spaces ===
    action_space = 21
    observation_space = 75

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
        60.0, 100.0, 60.0,  # left leg
        80.0, 30.0, 30.0,
        60.0, 100.0, 60.0,  # right leg
        80.0, 30.0, 30.0,
    ]

    # === FORCE WALK REWARDS ===
    # HUGE rewards for movement
    forward_speed_reward: float = 20.0     # MASSIVE reward for forward speed
    standing_penalty: float = -5.0         # BIG penalty for standing still
    alive_reward: float = 1.0              # Basic alive bonus

    # Smaller penalties
    action_penalty: float = 0.001           # Very small action penalty
    balance_penalty: float = 0.1            # Small balance penalty

    # Fall penalties
    fall_penalty: float = 20.0              # Big fall penalty

    # === Movement targets ===
    min_speed: float = 0.3                  # Minimum speed to avoid standing penalty
    target_speed: float = 1.0               # Target forward speed
    max_speed: float = 3.0                  # Maximum useful speed

    # === Termination ===
    termination_height: float = 0.5        # More forgiving
    max_roll_pitch: float = 1.5            # More forgiving angles

    # === Scaling ===
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1

    # === Debug ===
    enable_debug_arrows: bool = True
    debug_arrow_scale: float = 0.5

class HumanoidSNCForceWalkEnv(LocomotionEnv):
    """Environment that FORCES the robot to walk"""

    cfg: HumanoidSNCForceWalkEnvCfg

    def __init__(self, cfg: HumanoidSNCForceWalkEnvCfg, render_mode: str | None = None, **kwargs):
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

        # Debug arrows
        self._debug_draw = None
        if self.cfg.enable_debug_arrows:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                print("🚶 FORCE WALK debug arrows enabled!")
            except Exception as e:
                print(f"[DEBUG-ARROWS] unavailable: {e}")

        print("🚶 FORCE WALK Humanoid SNC Environment initialized!")
        print("   📢 This environment will FORCE the robot to walk!")

    def _get_observations(self) -> dict:
        """Enhanced observations for walking"""
        try:
            self._compute_intermediate_values()
        except Exception as e:
            print(f"[_get_observations] error: {e}")

        self._safe_fix_all()

        # Enhanced observations for walking
        parts = [
            # Position and height
            self.torso_position[:, 2:3],  # height

            # Velocity (IMPORTANT for walking)
            self.velocity,  # world velocity (3D) - robot needs to see its speed
            self.ang_velocity * self.cfg.angular_velocity_scale,  # angular velocity

            # Orientation (using sin/cos for continuity)
            torch.sin(self.roll).unsqueeze(-1),
            torch.cos(self.roll).unsqueeze(-1),
            torch.sin(self.pitch).unsqueeze(-1),
            torch.cos(self.pitch).unsqueeze(-1),
            torch.sin(self.yaw).unsqueeze(-1),
            torch.cos(self.yaw).unsqueeze(-1),

            # Joint states
            self.dof_pos_scaled,
            self.dof_vel * self.cfg.dof_vel_scale,

            # Current actions (for temporal consistency)
            self.actions,

            # Speed information (help robot understand its movement)
            torch.norm(self.velocity[:, :2], dim=1, keepdim=True),  # current XY speed
            torch.full((self.num_envs, 1), self.cfg.target_speed, device=self._torch_device),  # target speed
        ]

        obs = torch.cat(parts, dim=-1)

        # Ensure correct size
        if obs.shape[1] > self._num_obs:
            obs = obs[:, :self._num_obs]
        elif obs.shape[1] < self._num_obs:
            pad = torch.zeros((obs.shape[0], self._num_obs - obs.shape[1]), device=self._torch_device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

        # Safety
        obs = torch.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = torch.clamp(obs, min=-10.0, max=10.0)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """FORCE WALK reward function - huge penalties for not moving!"""

        self._safe_fix_all()

        # === 1. Forward Speed Reward (MASSIVE) ===
        forward_speed = self.velocity[:, 0]  # x-direction speed

        # HUGE reward for any forward movement
        speed_reward = self.cfg.forward_speed_reward * torch.clamp(
            forward_speed / self.cfg.target_speed,
            0.0, 2.0  # Cap at 2x target speed
        )

        # === 2. Standing Still Penalty (BIG) ===
        current_speed = torch.norm(self.velocity[:, :2], dim=1)  # XY speed
        standing_penalty = torch.where(
            current_speed < self.cfg.min_speed,
            torch.ones_like(current_speed) * self.cfg.standing_penalty,
            torch.zeros_like(current_speed)
        )

        # === 3. Alive Reward ===
        alive_reward = torch.ones(self.num_envs, device=self._torch_device) * self.cfg.alive_reward

        # === 4. Small Balance Penalty ===
        balance_penalty = -self.cfg.balance_penalty * (torch.abs(self.roll) + torch.abs(self.pitch))

        # === 5. Small Action Penalty ===
        action_penalty = -self.cfg.action_penalty * torch.mean(self.actions ** 2, dim=-1)

        # === 6. Height Reward (keep reasonable height) ===
        height_reward = torch.clamp((self.torso_position[:, 2] - 0.5) / 0.5, -1.0, 1.0)

        # === Total Reward ===
        total_reward = (
            speed_reward +
            standing_penalty +
            alive_reward +
            balance_penalty +
            action_penalty +
            height_reward
        )

        # Safety clamps
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=50.0, neginf=-50.0)
        total_reward = torch.clamp(total_reward, min=-100.0, max=100.0)

        # Fall penalty
        total_reward = torch.where(
            self.reset_terminated,
            torch.ones_like(total_reward) * (-self.cfg.fall_penalty),
            total_reward
        )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """More forgiving termination to allow learning"""
        self._compute_intermediate_values()
        self._safe_fix_all()

        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # More forgiving termination
        height_died = self.torso_position[:, 2] < self.cfg.termination_height
        balance_died = (
            (torch.abs(self.roll) > self.cfg.max_roll_pitch) |
            (torch.abs(self.pitch) > self.cfg.max_roll_pitch)
        )

        died = height_died | balance_died
        return died, time_out

    def _safe_fix_all(self):
        """Fix all potential issues"""
        vars_to_fix = [
            "torso_position", "velocity", "ang_velocity",
            "roll", "pitch", "yaw", "dof_pos_scaled", "dof_vel", "actions"
        ]

        for var_name in vars_to_fix:
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                if isinstance(var, torch.Tensor):
                    var = torch.nan_to_num(var, nan=0.0, posinf=1e3, neginf=-1e3)

                    if var_name == "torso_position":
                        var = torch.clamp(var, min=-100.0, max=100.0)
                    elif var_name in ["velocity", "ang_velocity"]:
                        var = torch.clamp(var, min=-20.0, max=20.0)
                    elif var_name in ["roll", "pitch", "yaw"]:
                        var = torch.clamp(var, min=-6.28, max=6.28)
                    elif var_name == "dof_pos_scaled":
                        var = torch.clamp(var, min=-3.0, max=3.0)
                    elif var_name == "dof_vel":
                        var = torch.clamp(var, min=-50.0, max=50.0)
                    elif var_name == "actions":
                        var = torch.clamp(var, min=-1.0, max=1.0)

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
        """Step with enhanced debug arrows"""
        # Fix action shape and clamp
        if actions.dim() == 1:
            actions = actions.view(1, -1)
        if actions.shape[0] == self._num_actions and actions.shape[1] == self.num_envs:
            actions = actions.T

        actions = torch.clamp(actions, min=-1.0, max=1.0)
        actions = torch.nan_to_num(actions, nan=0.0)

        obs_dict, rewards, terminated, truncated, info = super().step(actions)
        dones = terminated | truncated

        # Enhanced debug arrows (show velocity)
        if self._debug_draw is not None:
            try:
                P = self.torso_position.detach().cpu().numpy()
                V = self.velocity.detach().cpu().numpy()
                scale = float(self.cfg.debug_arrow_scale)
                self._debug_draw.clear_lines()

                n = min(64, P.shape[0])
                for i in range(n):
                    x0, y0, z0 = P[i].tolist()
                    x1, y1, z1 = (P[i] + V[i] * scale).tolist()

                    # Color based on speed (green = fast, red = slow)
                    speed = np.linalg.norm(V[i][:2])
                    if speed > 0.5:
                        color = (0.0, 1.0, 0.0, 1.0)  # Green - good speed
                    elif speed > 0.2:
                        color = (1.0, 1.0, 0.0, 1.0)  # Yellow - medium speed
                    else:
                        color = (1.0, 0.0, 0.0, 1.0)  # Red - too slow

                    self._debug_draw.draw_line([x0, y0, z0], [x1, y1, z1], color)
            except Exception:
                pass

        return obs_dict["policy"], rewards, dones, info