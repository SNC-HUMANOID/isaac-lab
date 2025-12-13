#!/usr/bin/env python3
import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Humanoid SNC Robot")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU instead of GPU")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_env import HumanoidSNCEnv, HumanoidSNCEnvCfg
from isaaclab_tasks.direct.humanoid_snc.agents.rsl_rl_ppo_cfg import HumanoidSNCPPORunnerCfg

def main():
    print("=" * 80)
    print("🤖 เริ่มต้นการเทรน Humanoid SNC (ปรับปรุงแล้ว)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    print(f"📱 Device: {device}")
    print(f"🌍 Environments: {args_cli.num_envs}")
    print(f"👁️  Headless: {args_cli.headless}")

    env_cfg = HumanoidSNCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    print("\n⚙️  การตั้งค่าสำคัญ:")
    print(f"   - Episode length: {env_cfg.episode_length_s}s")
    print(f"   - Action scale: {env_cfg.action_scale}")
    print(f"   - Decimation: {env_cfg.decimation}")
    print(f"   - Termination height: {env_cfg.termination_height}")

    print("\n🏗️  กำลังสร้าง environment...")
    env = HumanoidSNCEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    print(
        f"[DEBUG] num_envs={env.num_envs}, num_actions={env.num_actions}, "
        f"action_space.shape={env.action_space.shape}, obs_space.shape={env.observation_space.shape}"
    )

    agent_cfg = HumanoidSNCPPORunnerCfg()
    agent_cfg.seed = 42

    print("\n🧠 การตั้งค่า Training:")
    print(f"   - Max iterations: {agent_cfg.max_iterations}")
    print(f"   - Steps per env: {agent_cfg.num_steps_per_env}")
    print(f"   - Learning rate: {agent_cfg.algorithm.learning_rate}")
    print(f"   - Network: {agent_cfg.policy.actor_hidden_dims}")

    print("\n🚀 เริ่มต้น PPO Runner...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="logs", device=device)

    print("\n🎯 เป้าหมายการเทรน:")
    print("   1. เรียนรู้การยืนมั่นคง (iterations 0-500)")
    print("   2. พัฒนาการทรงตัว (iterations 500-1000)")
    print("   3. เริ่มต้นการเดิน (iterations 1000+)")

    print("\n" + "=" * 80)
    print("🔥 เริ่มต้นการเทรน!")
    print("=" * 80)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print("\n✅ การเทรนเสร็จสิ้นสำเร็จ!")
    except KeyboardInterrupt:
        print("\n⚠️  การเทรนถูกยกเลิกโดยผู้ใช้")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดระหว่างการเทรน: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()
        print("\n🏁 ปิดโปรแกรมเรียบร้อย")

if __name__ == "__main__":
    main()
