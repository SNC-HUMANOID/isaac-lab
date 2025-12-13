#!/usr/bin/env python3
"""
🚀 Improved Humanoid SNC Balance Training Script
Focus: Balance first, then walking
"""

import argparse
import torch
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Humanoid SNC Robot - Balance Focus")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
parser.add_argument("--stage", type=str, default="balance", choices=["balance", "walking"],
                   help="Training stage: balance or walking")
parser.add_argument("--load_model", type=str, default="", help="Path to pre-trained model for walking stage")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_balance_env import HumanoidSNCBalanceEnv, HumanoidSNCBalanceEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_balance_cfg import (
    HumanoidSNCBalancePPORunnerCfg,
    HumanoidSNCWalkingPPORunnerCfg
)

def create_balance_config():
    """สร้าง config สำหรับการเทรน balance"""
    cfg = HumanoidSNCBalanceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"

    # Focus on balance - ลด heading reward
    cfg.heading_weight = 0.1
    cfg.balance_weight = 3.0
    cfg.up_weight = 2.0
    cfg.episode_length_s = 3.0  # เริ่มสั้นๆ

    return cfg

def create_walking_config():
    """สร้าง config สำหรับการเทรน walking (หลังจากเรียนรู้ balance แล้ว)"""
    cfg = HumanoidSNCBalanceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"

    # Focus on walking - เพิ่ม heading reward
    cfg.heading_weight = 1.0
    cfg.balance_weight = 2.0
    cfg.up_weight = 1.5
    cfg.episode_length_s = 8.0  # ยาวขึ้นเพื่อการเดิน

    return cfg

def main():
    print("=" * 80)
    print(f"🤖 เทรน Humanoid SNC - Stage: {args_cli.stage.upper()}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")
    print(f"🎯 Stage: {args_cli.stage}")

    # Create environment based on stage
    if args_cli.stage == "balance":
        print("\n🎯 STAGE 1: Learning Balance")
        env_cfg = create_balance_config()
        agent_cfg = HumanoidSNCBalancePPORunnerCfg()
        log_dir = "logs/balance"
    else:
        print("\n🚶 STAGE 2: Learning Walking")
        env_cfg = create_walking_config()
        agent_cfg = HumanoidSNCWalkingPPORunnerCfg()
        log_dir = "logs/walking"

    print("\n⚙️  Environment Settings:")
    print(f"   - Episode length: {env_cfg.episode_length_s}s")
    print(f"   - Action scale: {env_cfg.action_scale}")
    print(f"   - Termination height: {env_cfg.termination_height}")
    print(f"   - Balance weight: {env_cfg.balance_weight}")
    print(f"   - Heading weight: {env_cfg.heading_weight}")

    print("\n🏗️  Creating environment...")
    env = HumanoidSNCBalanceEnv(cfg=env_cfg, render_mode="########################################################################
                       Learning iteration 7/2000                        

                       Computation: 200 steps/s (collection: 60.945s, learning 1.569s)
             Mean action noise std: 0.80
          Mean value_function loss: nan
               Mean surrogate loss: nan
                 Mean entropy loss: 25.1117
                       Mean reward: -inf
               Mean episode length: 1.87
--------------------------------------------------------------------------------
                   Total timesteps: 100032
                    Iteration time: 62.51s
                      Time elapsed: 00:04:31
                               ETA: 18:45:58

rgb_array" if args_cli.headless else None)

    print(f"[DEBUG] Environment created: {env.num_envs} envs, {env.num_actions} actions")
    print(f"        Observation space: {env.observation_space.shape}")
    print(f"        Action space: {env.action_space.shape}")

    # Agent configuration
    agent_cfg.seed = 42

    print(f"\n🧠 Training Configuration ({args_cli.stage}):")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Steps per env: {agent_cfg.num_steps_per_env}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate}")
    print(f"   - Network: {agent_cfg.policy.actor_hidden_dims}")
    print(f"   - Noise std: {agent_cfg.policy.init_noise_std}")

    print(f"\n🚀 Creating PPO Runner...")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    # Load pre-trained model for walking stage
    if args_cli.stage == "walking" and args_cli.load_model:
        if os.path.exists(args_cli.load_model):
            print(f"📂 Loading pre-trained model: {args_cli.load_model}")
            runner.load(args_cli.load_model)
        else:
            print(f"⚠️  Model file not found: {args_cli.load_model}")
            print("Starting from scratch...")

    print(f"\n🎯 Training Goals ({args_cli.stage}):")
    if args_cli.stage == "balance":
        print("   1. 🧘 Learn to stand upright (0-300 iterations)")
        print("   2. ⚖️  Maintain balance (300-600 iterations)")
        print("   3. 🏃 Small movements (600+ iterations)")
        print("   💡 เมื่อ balance ดีแล้ว ให้ใช้ --stage walking")
    else:
        print("   1. 🚶 Learn forward walking (0-500 iterations)")
        print("   2. 🏃 Improve gait stability (500-1000 iterations)")
        print("   3. 🎯 Target reaching (1000+ iterations)")

    print("\n" + "=" * 80)
    print(f"🔥 เริ่มเทรน Stage: {args_cli.stage.upper()}!")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"\n✅ Stage {args_cli.stage} training completed!")

        if args_cli.stage == "balance":
            print("\n🎉 Balance training completed!")
            print("🔄 Next steps:")
            print("   1. Check the training logs/balance/ directory")
            print("   2. Find the best model (highest reward)")
            print("   3. Run walking stage:")
            print(f"      ./isaaclab.sh -p train_humanoid_balance_improved.py --stage walking --load_model logs/balance/model_XXXX.pt --num_envs {args_cli.num_envs}")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print(f"\n🏁 Training session ended")

if __name__ == "__main__":
    main()