# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
_AGENTS_PKG = __name__ + ".agents"

gym.register(
    id="Isaac-Humanoid-SNC-Direct-v0",
    entry_point=f"{__name__}.humanoid_snc_env:HumanoidSNCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_snc_env:HumanoidSNCEnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_PKG}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_ppo_cfg:HumanoidSNCPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-SNC-R5-Direct-v0",
    entry_point=f"{__name__}.humanoid_snc_r5_env:HumanoidSNCR5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_snc_r5_env:HumanoidSNCR5EnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_PKG}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_ppo_r5_cfg:HumanoidSNCR5PPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-SNC-R5-Walking-Direct-v0",
    entry_point=f"{__name__}.humanoid_snc_r5_walking_env:HumanoidSNCR5WalkingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_snc_r5_walking_env:HumanoidSNCR5WalkingEnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_PKG}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_ppo_walking_cfg:HumanoidSNCR5WalkingPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-SNC-R5-Minimal-Direct-v0",
    entry_point=f"{__name__}.humanoid_snc_r5_minimal_env:HumanoidSNCR5MinimalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_snc_r5_minimal_env:HumanoidSNCR5MinimalEnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_PKG}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_ppo_walking_cfg:HumanoidSNCR5WalkingPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_ppo_cfg.yaml",
    },
)
