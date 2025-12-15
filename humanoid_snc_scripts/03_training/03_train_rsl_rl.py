#!/usr/bin/env python3
"""3️⃣ เทรน RL ด้วย RSL-RL อย่างถูกต้อง"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Humanoid SNC with RSL-RL")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Inline environment definition
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

# Robot config
HUMANOID_SNC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=300.0,
            velocity_limit=10.0,
            stiffness=80.0,
            damping=4.0,
        ),
    },
)

@configclass
class TestEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 75
    state_space = 0
    
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
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
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True
    )
    robot: ArticulationCfg = HUMANOID_SNC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Required for locomotion env
    joint_gears: list = [50.0] * 21
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    joints_at_limit_cost_scale: float = 0.1
    death_cost: float = -1.0
    termination_height: float = 0.31
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

class TestEnv(LocomotionEnv):
    cfg: TestEnvCfg
    def __init__(self, cfg: TestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

def main():
    # Create environment
    env_cfg = TestEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.decimation = 2
    env_cfg.sim.dt = 1.0 / 120.0
    
    print("🔧 Environment Configuration:")
    print_dict(env_cfg, nesting=2)
    
    env = TestEnv(cfg=env_cfg)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    print(f"🤖 Environment: {env.num_envs} robots")
    print(f"🎮 Action space: {env.action_space}")
    print(f"👁️  Observation space: {env.observation_space}")
    
    # Algorithm configuration - complete RSL-RL config
    agent_cfg = {
        # Runner settings
        "algorithm_class_name": "PPO",
        "policy_class_name": "ActorCritic", 
        "num_steps_per_env": 24,
        "max_iterations": args_cli.max_iterations,
        "save_interval": 50,
        "empirical_normalization": False,
        
        # Algorithm settings
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 1e-3,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
        },
        
        # Policy settings
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 0.1,
            "use_orthogonal_initialization": True,
        }
    }
    
    # Setup logging
    log_root_path = os.path.join("logs", "rsl_rl", "humanoid_snc")
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=str(env.device))
    
    print(f"🚀 Starting training:")
    print(f"   📊 Environments: {args_cli.num_envs}")
    print(f"   🔄 Max iterations: {args_cli.max_iterations}")
    print(f"   📁 Log directory: {log_dir}")
    
    # Save configs
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # Train
    try:
        runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
        print("✅ Training completed!")
    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user")
    
    print(f"📁 Logs saved to: {log_dir}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()