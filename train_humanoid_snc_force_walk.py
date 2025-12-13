#!/usr/bin/env python3
"""
🚶 FORCE WALK Humanoid Training
This FORCES the robot to walk by giving HUGE penalties for not moving!
"""

import argparse
import torch
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FORCE WALK Humanoid Training")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_force_walk_env import HumanoidSNCForceWalkEnv, HumanoidSNCForceWalkEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_balance_cfg import HumanoidSNCBalancePPORunnerCfg

def create_force_walk_agent():
    """Agent config for FORCE WALK training"""
    agent_cfg = HumanoidSNCBalancePPORunnerCfg()
    agent_cfg.experiment_name = "humanoid_force_walk"
    agent_cfg.max_iterations = 3000
    agent_cfg.seed = 42

    # AGGRESSIVE learning for fast walking
    agent_cfg.algorithm.learning_rate = 5e-4     # Higher learning rate
    agent_cfg.policy.init_noise_std = 0.8        # HIGH exploration for walking
    agent_cfg.num_steps_per_env = 64             # More experience per update
    agent_cfg.algorithm.entropy_coef = 0.03      # Good exploration
    agent_cfg.algorithm.clip_param = 0.2         # Standard clipping
    agent_cfg.algorithm.max_grad_norm = 1.0      # Standard gradient norm
    agent_cfg.algorithm.num_learning_epochs = 10 # More learning per batch
    agent_cfg.algorithm.num_mini_batches = 8     # Good batch size

    return agent_cfg

def main():
    print("=" * 80)
    print("🚶 FORCE WALK Humanoid Training - FORCES WALKING!")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")

    # Create FORCE WALK environment
    env_cfg = HumanoidSNCForceWalkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    agent_cfg = create_force_walk_agent()

    print(f"\n🚶 FORCE WALK Settings:")
    print(f"   - Forward speed reward: {env_cfg.forward_speed_reward} (MASSIVE!)")
    print(f"   - Standing penalty: {env_cfg.standing_penalty} (BIG PENALTY!)")
    print(f"   - Min speed threshold: {env_cfg.min_speed} m/s")
    print(f"   - Target speed: {env_cfg.target_speed} m/s")
    print(f"   - Episode length: {env_cfg.episode_length_s}s")
    print(f"   - Action scale: {env_cfg.action_scale} (FULL RANGE!)")

    print(f"\n🏗️  Creating FORCE WALK environment...")
    env = HumanoidSNCForceWalkEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(f"[INFO] FORCE WALK Environment created!")
    print(f"       - Environments: {env.num_envs}")
    print(f"       - Actions: {env.num_actions}")
    print(f"       - Observations: {env.observation_space.shape}")

    print(f"\n🚶 AGGRESSIVE Learning Parameters:")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate} (HIGH!)")
    print(f"   - Initial noise: {agent_cfg.policy.init_noise_std} (HIGH EXPLORATION!)")
    print(f"   - Steps per env: {agent_cfg.num_steps_per_env} (MORE DATA!)")
    print(f"   - Learning epochs: {agent_cfg.algorithm.num_learning_epochs} (MORE LEARNING!)")

    log_dir = f"logs/{agent_cfg.experiment_name}"
    print(f"\n🚀 Creating FORCE WALK PPO Runner...")
    print(f"   - Log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    print(f"\n🎯 FORCE WALK Strategy:")
    print("   1. 🚶 HUGE rewards for ANY forward movement (20x multiplier!)")
    print("   2. 🛑 BIG penalties for standing still (-5.0)")
    print("   3. 🎮 Full action range (1.0 scale)")
    print("   4. 🔍 HIGH exploration (0.8 noise)")
    print("   5. 📊 Real-time debug arrows (green=fast, red=slow)")
    print("   6. 🎯 Speed feedback in observations")

    print(f"\n💡 Expected FORCE WALK Results:")
    print("   - Robot MUST move forward or get huge penalty")
    print("   - Any forward speed gets massive reward")
    print("   - Should see walking attempts within 50 iterations")
    print("   - Debug arrows will show: GREEN=good speed, RED=too slow")
    print("   - Target: consistent forward movement > 0.3 m/s")

    print(f"\n📊 Watch These Metrics:")
    print("   - Mean reward: should be positive (lots of forward rewards)")
    print("   - Episode length: should increase as robot learns")
    print("   - Policy loss: should decrease steadily")
    print("   - Standing penalty should decrease over time")

    print("\n" + "=" * 80)
    print("🔥 Starting FORCE WALK Training! Robot MUST walk!")
    print("=== If you see RED arrows, robot is too slow! ===")
    print("=== GREEN arrows = SUCCESS! ===")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"\n✅ FORCE WALK training completed!")

        print(f"\n🎉 Final Results Check:")
        print("   - Did you see GREEN arrows? (good speed)")
        print("   - Was the robot moving forward consistently?")
        print("   - Were episode lengths increasing?")
        print("   - Was mean reward positive?")

        if args_cli.headless:
            print(f"\n🎬 To see the results visually, run:")
            print(f"   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py")
            print(f"   --task Isaac-Humanoid-SNC-Force-Walk-v0")
            print(f"   --load_run logs/humanoid_force_walk/[TIMESTAMP]/")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print(f"\n🏁 FORCE WALK training session ended")

if __name__ == "__main__":
    main()