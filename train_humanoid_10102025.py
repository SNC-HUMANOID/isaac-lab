#!/usr/bin/env python3

import argparse
from isaaclab.app import AppLauncher

# ===========================================
#               CLI ARGUMENTS
# ===========================================

parser = argparse.ArgumentParser(description="Train Humanoid 10102025 with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--task", type=str, default="Isaac-Humanoid-10102025-Direct-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=3000)

# ⬇️ IsaacLab will add --headless, --renderer etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ===========================================
#          START SIMULATION
# ===========================================

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================
#      IMPORTS AFTER SIM START
# ===========================================

import gymnasium as gym
import os
from isaaclab_tasks.direct.humanoid_10102025.humanoid_10102025_env import (
    Humanoid10102025Env,
    Humanoid10102025EnvCfg,
)
from isaaclab_tasks.humanoid_10102025.rsl_rl_ppo_cfg import Humanoid10102025PPORunnerCfg
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import torch


# ===========================================
#                  TRAIN
# ===========================================

def main():

    # ✅ Config ENV
    env_cfg = Humanoid10102025EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # ✅ Create ENV
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # ✅ Vec Wrapper
    env = RslRlVecEnvWrapper(env)

    # ✅ PPO Config
    agent_cfg = Humanoid10102025PPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations

    # Training log dir
    log_dir = "logs/humanoid_10102025/rsl_rl"
    os.makedirs(log_dir, exist_ok=True)

    device = env.unwrapped.device if torch.cuda.is_available() else "cpu"

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=device
    )

    print(f"🚀 Training Humanoid-10102025 | Device: {device} | Env: {args_cli.num_envs}")
    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
