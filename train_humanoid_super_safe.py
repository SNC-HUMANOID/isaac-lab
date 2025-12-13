#!/usr/bin/env python3
"""
🛡️ SUPER SAFE Humanoid Training
Absolutely no NaN, no explosions, just safe walking
"""

import argparse
import torch
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Super Safe Humanoid Training")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_simple_env import HumanoidSNCSimpleEnv, HumanoidSNCSimpleEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_balance_cfg import HumanoidSNCBalancePPORunnerCfg

def main():
    print("=" * 80)
    print("🛡️ SUPER SAFE Humanoid Training")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")

    # Create super safe environment
    env_cfg = HumanoidSNCSimpleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    print(f"\n🛡️ SUPER SAFE Settings:")
    print(f"   - Episode length: {env_cfg.episode_length_s}s")
    print(f"   - Action scale: {env_cfg.action_scale}")
    print(f"   - Alive reward: {env_cfg.alive_reward}")
    print(f"   - Forward reward: {env_cfg.forward_reward}")
    print(f"   - Target velocity: {env_cfg.target_velocity} m/s")
    print(f"   - Fall penalty: {env_cfg.fall_penalty}")

    print(f"\n🏗️  Creating SUPER SAFE environment...")
    env = HumanoidSNCSimpleEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(f"[INFO] SUPER SAFE Environment created!")
    print(f"       - Environments: {env.num_envs}")
    print(f"       - Actions: {env.num_actions}")
    print(f"       - Observations: {env.observation_space.shape}")

    # Super conservative agent config
    agent_cfg = HumanoidSNCBalancePPORunnerCfg()
    agent_cfg.experiment_name = "humanoid_super_safe"
    agent_cfg.max_iterations = 1000
    agent_cfg.seed = 42

    # ULTRA conservative parameters
    agent_cfg.algorithm.learning_rate = 1e-4     # Very low
    agent_cfg.policy.init_noise_std = 0.3        # Moderate exploration
    agent_cfg.num_steps_per_env = 16             # Small batches
    agent_cfg.algorithm.entropy_coef = 0.01      # Some exploration
    agent_cfg.algorithm.clip_param = 0.1         # Conservative clipping
    agent_cfg.algorithm.max_grad_norm = 0.5      # Prevent gradient explosion

    print(f"\n🛡️ SUPER SAFE Training Parameters:")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate}")
    print(f"   - Initial noise: {agent_cfg.policy.init_noise_std}")
    print(f"   - Clip param: {agent_cfg.algorithm.clip_param}")
    print(f"   - Max grad norm: {agent_cfg.algorithm.max_grad_norm}")

    log_dir = f"logs/{agent_cfg.experiment_name}"
    print(f"\n🚀 Creating SUPER SAFE PPO Runner...")
    print(f"   - Log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    print(f"\n🎯 SUPER SAFE Training Strategy:")
    print("   1. 🛡️ ALL variables clamped to prevent explosions")
    print("   2. 🧮 SIMPLE math only (no exponentials, no complex calculations)")
    print("   3. 🚶 Basic walking rewards (forward movement + staying up)")
    print("   4. 📉 Conservative learning parameters")
    print("   5. 🔍 Continuous NaN/inf monitoring and fixing")

    print(f"\n💡 Expected Results:")
    print("   - NO NaN or inf values")
    print("   - Rewards in range [-20, +20]")
    print("   - Gradual learning to walk forward")
    print("   - Episode length should increase")

    print("\n" + "=" * 80)
    print("🔥 Starting SUPER SAFE Training!")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"\n✅ SUPER SAFE training completed!")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print(f"\n🏁 SUPER SAFE training session ended")

if __name__ == "__main__":
    main()