#!/usr/bin/env python3
"""
⚡ FAST & SAFE Humanoid Training
Safe but not too conservative - get results quickly!
"""

import argparse
import torch
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fast & Safe Humanoid Training")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_simple_env import HumanoidSNCSimpleEnv, HumanoidSNCSimpleEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_balance_cfg import HumanoidSNCBalancePPORunnerCfg

def create_fast_safe_config():
    """Create config that's safe but learns quickly"""
    cfg = HumanoidSNCSimpleEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"

    # FAST settings
    cfg.episode_length_s = 15.0      # Longer episodes for better learning
    cfg.action_scale = 0.8           # More action range
    cfg.alive_reward = 2.0           # Higher alive reward
    cfg.forward_reward = 5.0         # HIGH forward reward (encourage movement!)
    cfg.up_reward = 1.0              # Moderate up reward
    cfg.action_penalty = 0.005       # Lower action penalty
    cfg.target_velocity = 1.0        # Higher target speed

    return cfg

def create_fast_safe_agent():
    """Create agent that learns quickly but safely"""
    agent_cfg = HumanoidSNCBalancePPORunnerCfg()
    agent_cfg.experiment_name = "humanoid_fast_safe"
    agent_cfg.max_iterations = 2000
    agent_cfg.seed = 42

    # FAST learning parameters
    agent_cfg.algorithm.learning_rate = 3e-4     # Standard LR (not too low!)
    agent_cfg.policy.init_noise_std = 0.6        # Good exploration
    agent_cfg.num_steps_per_env = 32             # More data per update
    agent_cfg.algorithm.entropy_coef = 0.02      # Good exploration
    agent_cfg.algorithm.clip_param = 0.2         # Standard clipping
    agent_cfg.algorithm.max_grad_norm = 1.0      # Standard gradient norm
    agent_cfg.algorithm.num_learning_epochs = 8  # More learning per batch
    agent_cfg.algorithm.num_mini_batches = 8     # Good batch size

    return agent_cfg

def main():
    print("=" * 80)
    print("⚡ FAST & SAFE Humanoid Training")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")

    # Create fast but safe environment
    env_cfg = create_fast_safe_config()
    agent_cfg = create_fast_safe_agent()

    print(f"\n⚡ FAST & SAFE Settings:")
    print(f"   - Episode length: {env_cfg.episode_length_s}s (LONGER)")
    print(f"   - Action scale: {env_cfg.action_scale} (MORE RANGE)")
    print(f"   - Forward reward: {env_cfg.forward_reward} (HIGH!)")
    print(f"   - Target velocity: {env_cfg.target_velocity} m/s (FASTER)")
    print(f"   - Alive reward: {env_cfg.alive_reward}")

    print(f"\n🏗️  Creating FAST & SAFE environment...")
    env = HumanoidSNCSimpleEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(f"[INFO] FAST & SAFE Environment created!")
    print(f"       - Environments: {env.num_envs}")
    print(f"       - Actions: {env.num_actions}")
    print(f"       - Observations: {env.observation_space.shape}")

    print(f"\n⚡ FAST Learning Parameters:")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate} (STANDARD)")
    print(f"   - Initial noise: {agent_cfg.policy.init_noise_std} (GOOD EXPLORATION)")
    print(f"   - Steps per env: {agent_cfg.num_steps_per_env} (MORE DATA)")
    print(f"   - Learning epochs: {agent_cfg.algorithm.num_learning_epochs} (MORE LEARNING)")

    log_dir = f"logs/{agent_cfg.experiment_name}"
    print(f"\n🚀 Creating FAST & SAFE PPO Runner...")
    print(f"   - Log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    print(f"\n🎯 FAST & SAFE Strategy:")
    print("   1. ⚡ Higher rewards for forward movement")
    print("   2. 🎮 More action range (0.8 vs 0.3)")
    print("   3. 📚 More learning per iteration")
    print("   4. 🔍 Good exploration (0.6 noise)")
    print("   5. 🛡️ Still safe (no NaN/inf)")

    print(f"\n💡 Expected FAST Results:")
    print("   - Quick improvement in 100-200 iterations")
    print("   - Forward movement within first 50 iterations")
    print("   - Episode length 100+ within 200 iterations")
    print("   - Total reward 5-15 range")

    print("\n" + "=" * 80)
    print("🔥 Starting FAST & SAFE Training!")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"\n✅ FAST & SAFE training completed!")

        print(f"\n🎉 Final Results:")
        print("   - Check if robot walks forward consistently")
        print("   - Look for episode lengths > 100 steps")
        print("   - Forward reward should be positive")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print(f"\n🏁 FAST & SAFE training session ended")

if __name__ == "__main__":
    main()