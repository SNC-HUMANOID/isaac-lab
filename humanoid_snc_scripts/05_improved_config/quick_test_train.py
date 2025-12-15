#!/usr/bin/env python3
"""🚀 Quick test version of Humanoid SNC training"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quick Test Humanoid SNC")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=5, help="Training iterations")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import numpy as np
from datetime import datetime

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Base environment
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

# Simple robot configuration
HUMANOID_SNC_SIMPLE_CFG = ArticulationCfg(
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
            stiffness=50.0,
            damping=5.0,
        ),
    },
)

@configclass  
class SimpleHumanoidEnvCfg(DirectRLEnvCfg):
    """Simple Humanoid SNC Environment Configuration"""
    
    # Environment settings
    episode_length_s = 5.0  # Short episode for testing
    decimation = 2
    action_scale = 0.2
    action_space = 21
    observation_space = 75
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
    robot = HUMANOID_SNC_SIMPLE_CFG
    robot.prim_path = "/World/envs/env_.*/Robot"
    
    # Joint gears (21 joints)
    joint_gears: list = [1.0] * 21

class SimpleHumanoidEnv(LocomotionEnv):
    """Simple Humanoid SNC Environment"""
    
    cfg: SimpleHumanoidEnvCfg

    def __init__(self, cfg: SimpleHumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

def quick_test():
    """Quick test to verify environment works"""
    
    print("🚀 Quick Test - Humanoid SNC Environment")
    print(f"📊 Environments: {args_cli.num_envs}")
    print(f"🔄 Test iterations: {args_cli.max_iterations}")
    
    # Create environment
    env_cfg = SimpleHumanoidEnvCfg()
    env = SimpleHumanoidEnv(cfg=env_cfg)
    
    print(f"🤖 Environment created successfully!")
    print(f"🎮 Action space: {env.action_space}")
    print(f"👁️  Observation space: {env.observation_space}")
    
    # Simple test loop
    print("🏃‍♂️ Running quick test...")
    
    for iteration in range(args_cli.max_iterations):
        print(f"Test iteration: {iteration}")
        
        # Reset environment
        obs, _ = env.reset()
        print(f"Observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")
        
        # Take a few steps
        for step in range(10):
            # Random actions
            actions = torch.randn(env.num_envs, 21, device=env.device) * 0.1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            
            if step == 0:
                print(f"Reward shape: {reward.shape}")
                print(f"Average reward: {reward.mean().item():.3f}")
            
            if terminated.any() or truncated.any():
                print(f"Episode ended at step {step}")
                break
        
        print(f"✅ Test iteration {iteration} completed")
    
    print("✅ Quick test completed successfully!")
    env.close()

def main():
    """Main function"""
    quick_test()

if __name__ == "__main__":
    main()
    simulation_app.close()