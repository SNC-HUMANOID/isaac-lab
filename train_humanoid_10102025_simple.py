#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple training script for Humanoid 10102025 robot."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Humanoid 10102025.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab_assets.robots.humanoid_10102025 import HUMANOID_10102025_CFG

##
# Environment configuration
##

@configclass
class Humanoid10102025EnvCfg(DirectRLEnvCfg):
    """Configuration for Humanoid 10102025 environment."""

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    # robot
    robot = HUMANOID_10102025_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # contact sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/pelvis"],
    )

    # environment settings
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 20  # 20 actuated joints
    observation_space = 74  # Base observations
    state_space = 0

    # reward scales
    lin_vel_reward_scale = 1.0
    ang_vel_reward_scale = 0.5
    joint_acc_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    alive_reward_scale = 2.0
    dof_at_limit_cost_scale = -1.0
    feet_air_time_reward_scale = 0.5

    # command ranges
    target_lin_vel_x_range = (0.0, 1.0)
    target_lin_vel_y_range = (-0.5, 0.5)
    target_ang_vel_z_range = (-1.0, 1.0)


class Humanoid10102025Env(DirectRLEnv):
    """Environment for Humanoid 10102025 robot."""

    cfg: Humanoid10102025EnvCfg

    def __init__(self, cfg: Humanoid10102025EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices
        self._joint_dof_idx, _ = self.scene["robot"].find_joints(".*")
        self._body_ids = [0]  # Pelvis body ID

        # Initialize buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # Command buffers
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_z]

        print("[INFO] Humanoid 10102025 environment initialized successfully!")

    def _setup_scene(self):
        """Setup the scene."""
        self.scene.articulations["robot"] = self.scene["robot"]
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-physics step."""
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions

    def _apply_action(self):
        """Apply actions to the robot."""
        self.scene["robot"].set_joint_position_target(self._processed_actions, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        """Get observations."""
        # Get robot state
        root_state = self.scene["robot"].data.root_state_w
        joint_pos = self.scene["robot"].data.joint_pos
        joint_vel = self.scene["robot"].data.joint_vel

        # Base observations
        obs = torch.cat(
            [
                root_state[:, :3],  # root position
                root_state[:, 3:7],  # root orientation (quat)
                root_state[:, 7:10],  # root linear velocity
                root_state[:, 10:13],  # root angular velocity
                joint_pos,  # joint positions
                joint_vel,  # joint velocities
                self._commands,  # velocity commands
                self._actions,  # previous actions
            ],
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Get robot state
        root_state = self.scene["robot"].data.root_state_w
        joint_acc = self.scene["robot"].data.joint_acc

        # Linear velocity tracking
        lin_vel_error = torch.sum(torch.square(root_state[:, 7:9] - self._commands[:, :2]), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / 0.25) * self.cfg.lin_vel_reward_scale

        # Angular velocity tracking
        ang_vel_error = torch.square(root_state[:, 12] - self._commands[:, 2])
        ang_vel_reward = torch.exp(-ang_vel_error / 0.25) * self.cfg.ang_vel_reward_scale

        # Joint acceleration penalty
        joint_acc_reward = torch.sum(torch.square(joint_acc), dim=1) * self.cfg.joint_acc_reward_scale

        # Action rate penalty
        action_rate_reward = torch.sum(torch.square(self._actions - self._previous_actions), dim=1) * self.cfg.action_rate_reward_scale

        # Alive reward
        alive_reward = torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward_scale

        # Total reward
        total_reward = lin_vel_reward + ang_vel_reward + joint_acc_reward + action_rate_reward + alive_reward

        # Store previous actions
        self._previous_actions[:] = self._actions

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get termination and truncation signals."""
        # Get robot state
        root_state = self.scene["robot"].data.root_state_w

        # Termination: robot falls
        terminated = torch.logical_or(root_state[:, 2] < 0.5, root_state[:, 2] > 2.0)

        # Truncation: episode length
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None:
            env_ids = self.scene["robot"]._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset robot state
        joint_pos = self.scene["robot"].data.default_joint_pos[env_ids]
        joint_vel = self.scene["robot"].data.default_joint_vel[env_ids]
        default_root_state = self.scene["robot"].data.default_root_state[env_ids]

        self.scene["robot"].write_root_state_to_sim(default_root_state, env_ids)
        self.scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset commands
        self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * \
            (self.cfg.target_lin_vel_x_range[1] - self.cfg.target_lin_vel_x_range[0]) + self.cfg.target_lin_vel_x_range[0]
        self._commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * \
            (self.cfg.target_lin_vel_y_range[1] - self.cfg.target_lin_vel_y_range[0]) + self.cfg.target_lin_vel_y_range[0]
        self._commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * \
            (self.cfg.target_ang_vel_z_range[1] - self.cfg.target_ang_vel_z_range[0]) + self.cfg.target_ang_vel_z_range[0]

        # Reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0


##
# Main training script
##

def main():
    """Main training function."""
    # Create environment configuration
    env_cfg = Humanoid10102025EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment
    env = Humanoid10102025Env(cfg=env_cfg)

    print(f"[INFO] Starting training with {env_cfg.scene.num_envs} environments")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    # Simple training loop (random actions for testing)
    for iteration in range(args_cli.max_iterations):
        # Random actions
        actions = 2.0 * torch.rand(env.num_envs, env_cfg.action_space, device=env.device) - 1.0

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        if iteration % 100 == 0:
            mean_reward = rewards.mean().item()
            print(f"[INFO] Iteration {iteration}: Mean reward = {mean_reward:.3f}")

    # Close environment
    env.close()

    print("[INFO] Training completed!")


if __name__ == "__main__":
    # Run main function
    main()
    # Close simulation
    simulation_app.close()
