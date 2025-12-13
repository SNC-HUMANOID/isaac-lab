# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .flat_env_cfg import Humanoid10102025FlatEnvCfg, Humanoid10102025FlatEnvCfg_PLAY
from .rough_env_cfg import Humanoid10102025RoughEnvCfg, Humanoid10102025RoughEnvCfg_PLAY

##
# Register Gym environments
##

gym.register(
    id="Isaac-Velocity-Flat-Humanoid-10102025-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Humanoid10102025FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Humanoid10102025PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Humanoid-10102025-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Humanoid10102025FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Humanoid10102025PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-10102025-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Humanoid10102025RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Humanoid10102025PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-10102025-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Humanoid10102025RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Humanoid10102025PPORunnerCfg",
    },
)
