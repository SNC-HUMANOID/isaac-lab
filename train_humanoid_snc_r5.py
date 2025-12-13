#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train Humanoid SNC R5 with reinforcement learning using RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--force_cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import isaac lab modules
import isaaclab
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Register all Isaac Lab environments
import isaaclab_tasks  # noqa: F401

# Import the humanoid SNC R5 environment configuration
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_r5_env import HumanoidSNCR5EnvCfg


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg = HumanoidSNCR5EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # create isaac environment
    env = gym.make("Isaac-Humanoid-SNC-R5-Direct-v0", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", "humanoid_snc_r5")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.seed is not None:
        log_dir += f"_{args_cli.seed}"

    # create runner configuration
    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.seed = args_cli.seed if args_cli.seed is not None else 42
    runner_cfg.experiment_name = "humanoid_snc_r5"
    runner_cfg.device = "cuda:0" if torch.cuda.is_available() and not args_cli.force_cpu else "cpu"

    # Algorithm configuration
    runner_cfg.algorithm.value_loss_coef = 1.0
    runner_cfg.algorithm.use_clipped_value_loss = True
    runner_cfg.algorithm.clip_param = 0.2
    runner_cfg.algorithm.entropy_coef = 0.01
    runner_cfg.algorithm.num_learning_epochs = 5
    runner_cfg.algorithm.num_mini_batches = 4
    runner_cfg.algorithm.learning_rate = 3.0e-4
    runner_cfg.algorithm.schedule = "adaptive"
    runner_cfg.algorithm.gamma = 0.99
    runner_cfg.algorithm.lam = 0.95
    runner_cfg.algorithm.desired_kl = 0.01
    runner_cfg.algorithm.max_grad_norm = 1.0

    # Training configuration
    runner_cfg.max_iterations = 1500
    runner_cfg.save_interval = 50
    runner_cfg.experiment_name = "humanoid_snc_r5"
    runner_cfg.run_name = ""
    runner_cfg.logger = "tensorboard"
    runner_cfg.neptune_project = "isaaclab"
    runner_cfg.wandb_project = "isaaclab"

    # Policy configuration
    runner_cfg.policy.class_name = "ActorCritic"
    runner_cfg.policy.init_noise_std = 1.0
    runner_cfg.policy.actor_hidden_dims = [512, 256, 128]
    runner_cfg.policy.critic_hidden_dims = [512, 256, 128]
    runner_cfg.policy.activation = "elu"

    print(f"[INFO] RSL-RL configuration:")
    print_dict(runner_cfg, nesting=4)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)

    # set seed of the env
    env.seed(runner_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), runner_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), runner_cfg)

    # run training
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    # save the final policy
    export_policy_as_jit(
        runner.alg.actor_critic,
        path=os.path.join(log_root_path, log_dir),
        filename="policy.pt"
    )
    export_policy_as_onnx(
        runner.alg.actor_critic,
        path=os.path.join(log_root_path, log_dir),
        filename="policy.onnx",
        verbose=False,
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()