# SPDX-License-Identifier: BSD-3-Clause
"""
Gym registration สำหรับ Humanoid SNC
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Humanoid-SNC-Direct-v0",
    entry_point=f"{__name__}.humanoid_snc_env:HumanoidSNCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_snc_env:HumanoidSNCEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidSNCPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
