#!/usr/bin/env python3
"""🚀 Minimal working Humanoid SNC training script"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal Humanoid SNC Training")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=50, help="Training iterations")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Simple robot configuration
HUMANOID_SNC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=5.0,
        ),
    },
)

@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Simple Humanoid Environment Configuration"""
    
    # Environment settings
    episode_length_s = 10.0
    decimation = 2
    action_scale = 0.3
    action_space = 21
    observation_space = 21  # Simple observation space
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
    )
    
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True
    )

    # Robot
    robot = HUMANOID_SNC_CFG
    robot.prim_path = "/World/envs/env_.*/Robot"

class HumanoidEnv(DirectRLEnv):
    """Simple Humanoid Environment"""
    
    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_observations(self) -> dict:
        """Simple observation - just joint positions"""
        obs = self.robot.data.joint_pos
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Simple reward - staying upright"""
        # Reward for staying upright (z-axis position)
        height_reward = self.robot.data.root_pos_w[:, 2] - 0.8  # Target height 1.0m, floor at 0.0m
        
        # Small penalty for large joint angles
        joint_penalty = -0.1 * torch.sum(torch.abs(self.robot.data.joint_pos), dim=1)
        
        return height_reward + joint_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Episode termination conditions"""
        # Terminate if robot falls below certain height
        height_termination = self.robot.data.root_pos_w[:, 2] < 0.3
        
        # Time-based truncation
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        return height_termination, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments"""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        # Reset robot state
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        
        # Reset joint positions and velocities
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        default_joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)

def minimal_train():
    """Minimal training function"""
    
    print("🚀 Minimal Humanoid SNC Training")
    print(f"📊 Environments: {args_cli.num_envs}")
    print(f"🔄 Max iterations: {args_cli.max_iterations}")
    
    # Create environment
    env_cfg = HumanoidEnvCfg()
    env = HumanoidEnv(cfg=env_cfg)
    
    print(f"🤖 Environment created with {env.num_envs} robots")
    print(f"🎮 Action space: {env.action_space}")
    print(f"👁️  Observation space: {env.observation_space}")
    
    # Simple random policy
    print("🏃‍♂️ Starting training...")
    
    for iteration in range(args_cli.max_iterations):
        episode_reward = 0.0
        
        # Reset environment
        obs, _ = env.reset()
        
        # Episode loop
        for step in range(int(env_cfg.episode_length_s * 30)):  # 30 Hz
            # Random actions (small values)
            actions = torch.randn(env.num_envs, 21, device=env.device) * 0.1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            
            episode_reward += reward.mean().item()
            
            if terminated.any() or truncated.any():
                break
        
        if iteration % 10 == 0:
            print(f"⏱️  Iteration: {iteration:4d}, Avg Reward: {episode_reward/max(step, 1):.3f}")
    
    print("✅ Training completed!")
    env.close()

def main():
    """Main function"""
    minimal_train()

if __name__ == "__main__":
    main()
    simulation_app.close()