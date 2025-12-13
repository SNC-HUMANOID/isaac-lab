#!/usr/bin/env python3
"""
🚀 Humanoid SNC with G1-Style Training
Best of both worlds: SNC robot + G1 proven training methods
"""

import argparse
import torch
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Humanoid SNC with G1-style rewards and parameters")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
parser.add_argument("--config", type=str, default="standard",
                   choices=["standard", "flat", "advanced"],
                   help="Training configuration: standard, flat, or advanced")
parser.add_argument("--target_speed", type=float, default=1.0, help="Target walking speed (m/s)")
parser.add_argument("--load_model", type=str, default="", help="Path to pre-trained model")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_g1_style_env import HumanoidSNCG1StyleEnv, HumanoidSNCG1StyleEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_g1_style_cfg import (
    HumanoidSNCG1StylePPORunnerCfg,
    HumanoidSNCG1FlatStylePPORunnerCfg,
    HumanoidSNCG1AdvancedPPORunnerCfg
)

def create_environment_config():
    """Create G1-style environment configuration"""
    cfg = HumanoidSNCG1StyleEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"

    # Set target velocity
    cfg.target_lin_vel_x = args_cli.target_speed
    cfg.target_lin_vel_y = 0.0
    cfg.target_ang_vel_z = 0.0

    return cfg

def get_agent_config():
    """Get agent configuration based on selected config"""
    if args_cli.config == "flat":
        return HumanoidSNCG1FlatStylePPORunnerCfg()
    elif args_cli.config == "advanced":
        return HumanoidSNCG1AdvancedPPORunnerCfg()
    else:
        return HumanoidSNCG1StylePPORunnerCfg()

def main():
    print("=" * 80)
    print("🤖 Humanoid SNC + G1 Style Training")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")
    print(f"⚙️  Config: {args_cli.config}")
    print(f"🏃 Target Speed: {args_cli.target_speed} m/s")

    # Create environment with G1-style config
    env_cfg = create_environment_config()
    agent_cfg = get_agent_config()

    print(f"\n🎯 G1-Style Configuration:")
    print(f"   - Episode length: {env_cfg.episode_length_s}s")
    print(f"   - Termination penalty: {env_cfg.termination_penalty}")
    print(f"   - Track velocity weight: {env_cfg.track_lin_vel_xy_weight}")
    print(f"   - Feet air time weight: {env_cfg.feet_air_time_weight}")
    print(f"   - Action rate penalty: {env_cfg.action_rate_weight}")

    print(f"\n🏗️  Creating G1-style environment...")
    env = HumanoidSNCG1StyleEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(f"[INFO] Environment created successfully!")
    print(f"       - Environments: {env.num_envs}")
    print(f"       - Actions: {env.num_actions}")
    print(f"       - Observations: {env.observation_space.shape}")

    # Agent configuration
    agent_cfg.seed = 42

    print(f"\n🧠 G1-Style Training Parameters:")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Steps per env: {agent_cfg.num_steps_per_env}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate}")
    print(f"   - Network size: {agent_cfg.policy.actor_hidden_dims}")
    print(f"   - Initial noise: {agent_cfg.policy.init_noise_std}")
    print(f"   - Entropy coef: {agent_cfg.algorithm.entropy_coef}")

    log_dir = f"logs/{agent_cfg.experiment_name}"
    print(f"\n🚀 Creating PPO Runner...")
    print(f"   - Log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    # Load pre-trained model if specified
    if args_cli.load_model and os.path.exists(args_cli.load_model):
        print(f"📂 Loading pre-trained model: {args_cli.load_model}")
        runner.load(args_cli.load_model)
    elif args_cli.load_model:
        print(f"⚠️  Model file not found: {args_cli.load_model}")

    print(f"\n🎯 G1-Style Training Goals:")
    print("   1. 🎯 Velocity tracking (primary objective)")
    print("   2. 🦶 Proper gait patterns")
    print("   3. ⚖️  Balance and stability")
    print("   4. 🎮 Smooth actions")
    print("   5. 📈 Continuous improvement")

    print(f"\n💡 Expected G1-Style Results:")
    print("   - Track lin vel xy: positive (velocity following)")
    print("   - Track ang vel z: positive (turning control)")
    print("   - Feet air time: positive (proper gait)")
    print("   - Termination penalty: -200 (when falling)")
    print("   - Total reward: progressive improvement")

    print("\n" + "=" * 80)
    print("🔥 Starting G1-Style Training!")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"\n✅ G1-style training completed successfully!")

        print(f"\n🎉 Training Summary:")
        print(f"   - Configuration: {args_cli.config}")
        print(f"   - Target speed: {args_cli.target_speed} m/s")
        print(f"   - Total iterations: {agent_cfg.max_iterations}")
        print(f"   - Model saved in: {log_dir}")

        print(f"\n🔄 Next Steps:")
        print("   1. Check training curves in TensorBoard")
        print("   2. Test the trained policy with play script")
        print("   3. Experiment with different target speeds")
        print("   4. Try advanced configuration for better performance")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print(f"\n🏁 G1-style training session ended")

if __name__ == "__main__":
    main()