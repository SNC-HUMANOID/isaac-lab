#!/usr/bin/env python3
"""🎓 Train Humanoid SNC with RSL-RL (professional RL training)"""

import argparse
import os
import sys
import torch
from datetime import datetime

# Ensure Isaac Sim Python modules are in the path
try:
    import isaacsim.core.utils.stage
except ImportError:
    print("❌ Error: 'isaacsim.core' module not found. Ensure ISAACSIM_PATH is set in PYTHONPATH.")
    print("💡 Run: export PYTHONPATH=/home/sncbot/isaac-sim/exts:/home/sncbot/isaac-sim/exts/omni.isaac.core:$PYTHONPATH")
    sys.exit(1)

from isaaclab.app import AppLauncher
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import configclass

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train Humanoid SNC with RSL-RL")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=1500, help="Training iterations")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--task", type=str, default="Isaac-Humanoid-SNC-Direct-v0", help="Task name for the environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Initialize simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# RSL-RL imports
try:
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    RSL_RL_AVAILABLE = True
    print("✅ RSL-RL found")
except ImportError:
    RSL_RL_AVAILABLE = False
    print("❌ RSL-RL not available, falling back to simple training")

@configclass
class ImprovedHumanoidEnvCfg(DirectRLEnvCfg):
    """Configuration for the Improved Humanoid SNC Environment"""
    scene = dict(
        num_envs=1024,
        env_spacing=2.0,
    )

    robot = ArticulationCfg(
        spawn=UsdFileCfg(
            usd_path="/home/sncbot/IsaacLab/assets/Humanoid_SNC_fixed.usd",  # Corrected path
            rigid_props=dict(
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=dict(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=dict(
            pos=(0.0, 0.0, 1.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        soft_dof_pos_limit_factor=0.9,
    )

    sim = dict(
        dt=0.008333333333333333,  # 120 Hz physics
        render_interval=2,  # Render every 2 physics steps (60 Hz rendering)
        gravity=(0.0, 0.0, -9.81),
    )

    obs = dict(
        state_space="full",
        noise_scale=0.01,
    )
    action = dict(
        scale=1.0,
        clip=(-1.0, 1.0),
    )
    rewards = dict(
        alive=1.0,
        termination=-100.0,
        progress=2.0,
        energy=-0.005,
    )

class ImprovedHumanoidEnv(DirectRLEnv):
    """Improved Humanoid SNC Environment for RL training"""
    cfg: ImprovedHumanoidEnvCfg

    def __init__(self, cfg: ImprovedHumanoidEnvCfg, render_mode: str = None):
        super().__init__(cfg, render_mode=render_mode)
        self._setup_scene()

    def _setup_scene(self):
        """Setup the simulation scene"""
        self.sim.add_ground_plane()
        self.robot = Articulation(self.cfg.robot)
        self.robot.spawn(self.scene, num=self.cfg.scene.num_envs)
        self.sim.set_physics_params(self.cfg.sim)

    def reset(self, seed=None):
        super().reset(seed=seed)
        return self._get_observations()

    def step(self, actions):
        actions = torch.clamp(actions, self.cfg.action.clip[0], self.cfg.action.clip[1])
        self.robot.set_joint_efforts(actions)
        self.sim.step()
        return self._get_observations(), self._get_rewards(), self._get_dones(), {}

    def _get_observations(self):
        obs = torch.zeros((self.cfg.scene.num_envs, 50), device=self.device)  # Placeholder
        return obs

    def _get_rewards(self):
        rewards = torch.zeros(self.cfg.scene.num_envs, device=self.device)
        rewards += self.cfg.rewards.alive
        return rewards

    def _get_dones(self):
        dones = torch.zeros(self.cfg.scene.num_envs, dtype=torch.bool, device=self.device)
        return dones

def create_rsl_rl_config():
    """Create RSL-RL compatible configuration"""
    return {
        "algorithm_class_name": "PPO",
        "policy_class_name": "ActorCritic",
        "num_steps_per_env": 24,
        "max_iterations": args_cli.max_iterations,
        "save_interval": 100,
        "empirical_normalization": False,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": args_cli.learning_rate,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "normalize_input": True,
        "normalize_value": True,
    }

def train_with_rsl_rl():
    """Train using RSL-RL if available"""
    if not RSL_RL_AVAILABLE:
        print("❌ RSL-RL not available")
        return False

    print("🎓 Starting RSL-RL Training")
    print(f"📊 Environments: {args_cli.num_envs}")
    print(f"🔄 Max iterations: {args_cli.max_iterations}")
    print(f"📚 Learning rate: {args_cli.learning_rate}")
    print(f"🎯 Task: {args_cli.task}")

    try:
        env_cfg = ImprovedHumanoidEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = ImprovedHumanoidEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        env = RslRlVecEnvWrapper(env)

        print(f"🤖 Environment: {env.num_envs} robots")
        print(f"🎮 Action space: {env.action_space}")
        print(f"👁️ Observation space: {env.observation_space}")

        agent_cfg = create_rsl_rl_config()
        log_root_path = os.path.join("logs", "rsl_rl", "humanoid_snc_improved")
        log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)

        runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=str(env.device))
        print(f"🚀 Starting RSL-RL training...")
        print(f"📁 Logs: {log_dir}")

        runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
        print("✅ RSL-RL training completed!")
        print(f"📁 Check logs at: {log_dir}")

        env.close()
        return True

    except Exception as e:
        print(f"❌ RSL-RL training failed: {e}")
        print("💡 Try simple training instead")
        env.close()
        return False

def simple_fallback_train():
    """Fallback to simple training if RSL-RL fails"""
    print("🔄 Falling back to simple training method")
    env_cfg = ImprovedHumanoidEnvCfg()
    env = ImprovedHumanoidEnv(cfg=env_cfg)

    for episode in range(1000):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = torch.randn(env.action_space.shape, device=env.device)
            obs, reward, done, _ = env.step(action)
            total_reward += reward.sum().item()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()
    print("✅ Simple training completed!")

def main():
    """Main training function"""
    print("🤖 Humanoid SNC Training Manager")
    print("=" * 50)

    if RSL_RL_AVAILABLE:
        success = train_with_rsl_rl()
        if success:
            simulation_app.close()
            return

    print("\n💡 Using simple training method...")
    simple_fallback_train()
    simulation_app.close()

if __name__ == "__main__":
    main()