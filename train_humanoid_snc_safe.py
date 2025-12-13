#!/usr/bin/env python3
"""
🛡️ Safe Humanoid SNC Training
Simplified and robust training script
"""

import argparse
import torch
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Safe Humanoid SNC Training")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_balance_env import HumanoidSNCBalanceEnv, HumanoidSNCBalanceEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_balance_cfg import HumanoidSNCBalancePPORunnerCfg

def main():
    print("=" * 80)
    print("🛡️ Safe Humanoid SNC Training")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")

    # Use stable balance environment
    env_cfg = HumanoidSNCBalanceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    # Conservative settings to prevent NaN
    env_cfg.episode_length_s = 5.0  # Short episodes
    env_cfg.action_scale = 0.3      # Very conservative
    env_cfg.balance_weight = 1.0    # Moderate balance focus
    env_cfg.up_weight = 0.5         # Moderate up weight

    print(f"\n🛡️ Safe Configuration:")
    print(f"   - Episode length: {env_cfg.episode_length_s}s")
    print(f"   - Action scale: {env_cfg.action_scale}")
    print(f"   - Balance weight: {env_cfg.balance_weight}")
    print(f"   - Termination height: {env_cfg.termination_height}")

    print(f"\n🏗️  Creating safe environment...")
    env = HumanoidSNCBalanceEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(f"[INFO] Safe Environment created!")
    print(f"       - Environments: {env.num_envs}")
    print(f"       - Actions: {env.num_actions}")
    print(f"       - Observations: {env.observation_space.shape}")

    # Conservative agent config
    agent_cfg = HumanoidSNCBalancePPORunnerCfg()
    agent_cfg.experiment_name = "humanoid_snc_safe"
    agent_cfg.max_iterations = 1000  # Shorter training
    agent_cfg.seed = 42

    # Conservative learning parameters
    agent_cfg.algorithm.learning_rate = 5e-5  # Very low LR
    agent_cfg.policy.init_noise_std = 0.05    # Low noise
    agent_cfg.num_steps_per_env = 16          # Smaller batches

    print(f"\n🛡️ Safe Training Parameters:")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate}")
    print(f"   - Initial noise: {agent_cfg.policy.init_noise_std}")
    print(f"   - Steps per env: {agent_cfg.num_steps_per_env}")

    log_dir = f"logs/{agent_cfg.experiment_name}"
    print(f"\n🚀 Creating safe PPO Runner...")
    print(f"   - Log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    print(f"\n🎯 Safe Training Goals:")
    print("   1. 🛡️ Stable training (no NaN/inf)")
    print("   2. 🧘 Learn basic balance")
    print("   3. ⚖️ Gradual improvement")
    print("   4. 📈 Consistent rewards")

    print("\n" + "=" * 80)
    print("🔥 Starting Safe Training!")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"\n✅ Safe training completed!")

    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print(f"\n🏁 Safe training session ended")

if __name__ == "__main__":
    main()