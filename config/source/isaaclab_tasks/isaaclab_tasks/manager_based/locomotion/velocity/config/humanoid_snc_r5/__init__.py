# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Humanoid SNC R5 locomotion environment."""

import gymnasium as gym

from . import agents
# REMOVED: LeftHipInverterWrapper - URDF now has correct left hip axis

##
# Register Gym environments.
##

def _make_flat_env(**kwargs):
    """Factory function to create flat environment."""
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(**kwargs)
    # No wrapper needed - URDF axis is now correct
    return env

gym.register(
    id="Isaac-Velocity-Flat-Humanoid-SNC-R5-v0",
    entry_point=_make_flat_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HumanoidSNCR5FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidSNCR5PPORunnerCfg",
    },
)

def _make_flat_play_env(**kwargs):
    """Factory function to create flat play environment."""
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(**kwargs)
    return env

def _make_rough_env(**kwargs):
    """Factory function to create rough environment."""
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(**kwargs)
    return env

def _make_rough_play_env(**kwargs):
    """Factory function to create rough play environment."""
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(**kwargs)
    return env

gym.register(
    id="Isaac-Velocity-Flat-Humanoid-SNC-R5-Play-v0",
    entry_point=_make_flat_play_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HumanoidSNCR5FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidSNCR5PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-SNC-R5-v0",
    entry_point=_make_rough_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HumanoidSNCR5RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidSNCR5PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-SNC-R5-Play-v0",
    entry_point=_make_rough_play_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HumanoidSNCR5RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidSNCR5PPORunnerCfg",
    },
)