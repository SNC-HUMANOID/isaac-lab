#!/usr/bin/env python3
"""
🚶 Simple Humanoid Movement Training
Focus on getting the robot to move and walk, not just balance
"""

import argparse
import torch
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple Movement Training for Humanoid SNC")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
parser.add_argument("--speed", type=float, default=0.5, help="Target speed")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_simple_env import HumanoidSNCSimpleEnv, HumanoidSNCSimpleEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_balance_cfg import HumanoidSNCBalancePPORunnerCfg

def create_movement_config():
    """Create config focused on movement"""
    cfg = HumanoidSNCSimpleEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"

    # Movement-focused settings
    cfg.episode_length_s = 12.0  # Medium episodes for movement
    cfg.action_scale = 0.7       # More action range than default
    cfg.alive_reward = 1.5       # Good alive reward
    cfg.forward_reward = 8.0     # HIGH forward reward (encourage movement!)
    cfg.up_reward = 0.5          # Small up reward (don't be too strict)
    cfg.action_penalty = 0.003   # Very small action penalty
    cfg.target_velocity = args_cli.speed  # User-specified target speed
    cfg.termination_height = 0.5   # Forgiving termination

    return cfg

def create_movement_agent():
    """Create agent config for movement"""
    agent_cfg = HumanoidSNCBalancePPORunnerCfg()
    agent_cfg.experiment_name = "humanoid_simple_movement"
    agent_cfg.max_iterations = 1500
    agent_cfg.seed = 42

    # Movement-focused training
    agent_cfg.algorithm.learning_rate = 4e-4  # Good learning rate
    agent_cfg.policy.init_noise_std = 0.7     # High exploration
    agent_cfg.num_steps_per_env = 48          # Good experience
    agent_cfg.algorithm.entropy_coef = 0.025  # Good exploration
    agent_cfg.algorithm.clip_param = 0.2      # Standard clipping
    agent_cfg.algorithm.max_grad_norm = 1.0   # Standard grad norm
    agent_cfg.algorithm.num_learning_epochs = 8  # Moderate learning
    agent_cfg.algorithm.num_mini_batches = 8     # Good batch size

    return agent_cfg

def main():
    print("=" * 80)
    print("🚶 Simple Humanoid Movement Training")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")
    print(f"🏃 Target Speed: {args_cli.speed} m/s")

    # Create movement-focused environment
    env_cfg = create_movement_config()
    agent_cfg = create_movement_agent()

    print(f"\n🚶 Movement-Focused Settings:")
    print(f"   - Episode length: {env_cfg.episode_length_s}s")
    print(f"   - Action scale: {env_cfg.action_scale}")
    print(f"   - Forward reward: {env_cfg.forward_reward} (HIGH!)")
    print(f"   - Up reward: {env_cfg.up_reward} (moderate)")
    print(f"   - Alive reward: {env_cfg.alive_reward} (good)")
    print(f"   - Action penalty: {env_cfg.action_penalty} (LOW)")
    print(f"   - Target speed: {env_cfg.target_velocity} m/s")

    print(f"\n🏗️  Creating movement environment...")
    env = HumanoidSNCSimpleEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(f"[INFO] Movement Environment created!")
    print(f"       - Environments: {env.num_envs}")
    print(f"       - Actions: {env.num_actions}")
    print(f"       - Observation space: {env.observation_space.shape}")

    print(f"\n🧠 Movement Training Parameters:")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate}")
    print(f"   - Initial noise: {agent_cfg.policy.init_noise_std} (HIGH exploration!)")
    print(f"   - Entropy coef: {agent_cfg.algorithm.entropy_coef}")

    log_dir = f"logs/{agent_cfg.experiment_name}"
    print(f"\n🚀 Creating movement PPO Runner...")
    print(f"   - Log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    print(f"\n🎯 Movement Training Strategy:")
    print("   1. 🏃 HIGH forward reward → encourages forward movement")
    print("   2. 💰 Good alive reward → encourages not falling")
    print("   3. 🔍 HIGH exploration → tries different movements")
    print("   4. 🎮 LOW action penalty → allows movement")
    print("   5. 🛡️ Safe math → prevents NaN/inf issues")

    print(f"\n💡 What to expect:")
    print("   - Robot should start moving forward quickly")
    print("   - May fall but will learn to stay up while moving")
    print("   - Reward should increase as it learns to walk")
    print("   - Episode length should increase over time")

    print("\n" + "=" * 80)
    print("🔥 Starting Movement Training!")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"\n✅ Movement training completed!")

        print(f"\n🎉 Training Summary:")
        print("   - Check if robot learned to move forward")
        print("   - Look for increasing episode lengths")
        print("   - Check forward reward improvements")
        print("   - Watch for debug arrows showing movement")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print(f"\n🏁 Movement training session ended")

if __name__ == "__main__":
    main()