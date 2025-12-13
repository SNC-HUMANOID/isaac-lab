#!/usr/bin/env python3
"""
🎯 Humanoid SNC with EXACT G1 Training
Uses exact same reward functions and parameters as G1
"""

import argparse
import torch
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Humanoid SNC with EXACT G1 rewards")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_g1_exact_env import HumanoidSNCG1ExactEnv, HumanoidSNCG1ExactEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_g1_style_cfg import HumanoidSNCG1StylePPORunnerCfg

def main():
    print("=" * 80)
    print("🎯 Humanoid SNC with EXACT G1 Rewards")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")

    # Create EXACT G1 environment
    env_cfg = HumanoidSNCG1ExactEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    print(f"\n🎯 EXACT G1 Reward Weights:")
    print(f"   - termination_penalty: {env_cfg.termination_penalty}")
    print(f"   - track_lin_vel_xy_exp: {env_cfg.track_lin_vel_xy_exp_weight}")
    print(f"   - track_ang_vel_z_exp: {env_cfg.track_ang_vel_z_exp_weight}")
    print(f"   - lin_vel_z_l2: {env_cfg.lin_vel_z_l2_weight}")
    print(f"   - ang_vel_xy_l2: {env_cfg.ang_vel_xy_l2_weight}")
    print(f"   - dof_torques_l2: {env_cfg.dof_torques_l2_weight}")
    print(f"   - dof_acc_l2: {env_cfg.dof_acc_l2_weight}")
    print(f"   - action_rate_l2: {env_cfg.action_rate_l2_weight}")
    print(f"   - feet_air_time: {env_cfg.feet_air_time_weight}")
    print(f"   - flat_orientation_l2: {env_cfg.flat_orientation_l2_weight}")

    print(f"\n🏗️  Creating EXACT G1 environment...")
    env = HumanoidSNCG1ExactEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(f"[INFO] EXACT G1 Environment created!")
    print(f"       - Environments: {env.num_envs}")
    print(f"       - Actions: {env.num_actions}")
    print(f"       - Observations: {env.observation_space.shape}")

    # Use G1-style agent config
    agent_cfg = HumanoidSNCG1StylePPORunnerCfg()
    agent_cfg.experiment_name = "humanoid_snc_g1_exact"
    agent_cfg.seed = 42

    print(f"\n🧠 EXACT G1 Training Parameters:")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Steps per env: {agent_cfg.num_steps_per_env}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate}")
    print(f"   - Initial noise: {agent_cfg.policy.init_noise_std}")
    print(f"   - Network: {agent_cfg.policy.actor_hidden_dims}")

    log_dir = f"logs/{agent_cfg.experiment_name}"
    print(f"\n🚀 Creating PPO Runner...")
    print(f"   - Log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    print(f"\n🎯 Expected Results (like G1):")
    print("Episode_Reward/track_lin_vel_xy_exp: ~0.0087")
    print("Episode_Reward/track_ang_vel_z_exp: ~0.0027")
    print("       Episode_Reward/lin_vel_z_l2: ~-0.0021")
    print("      Episode_Reward/ang_vel_xy_l2: ~-0.0413")
    print("     Episode_Reward/dof_torques_l2: ~-0.0020")
    print("         Episode_Reward/dof_acc_l2: ~-0.0045")
    print("     Episode_Reward/action_rate_l2: ~-0.0158")
    print("      Episode_Reward/feet_air_time: ~0.0003")
    print("Episode_Reward/flat_orientation_l2: ~-0.0129")
    print("     Episode_Reward/dof_pos_limits: ~-0.0004")
    print("Episode_Reward/termination_penalty: ~-0.2000")

    print("\n" + "=" * 80)
    print("🔥 Starting EXACT G1 Training!")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"\n✅ EXACT G1 training completed!")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print(f"\n🏁 EXACT G1 training session ended")

if __name__ == "__main__":
    main()