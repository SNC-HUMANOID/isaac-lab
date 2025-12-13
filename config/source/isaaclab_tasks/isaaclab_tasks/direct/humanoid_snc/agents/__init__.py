# SPDX-License-Identifier: BSD-3-Clause
from .rsl_rl_ppo_cfg import HumanoidSNCPPORunnerCfg
try:
    from importlib.resources import files as _files
    SKRL_PPO_CFG_PATH = str(_files(__package__) / "skrl_ppo_cfg.yaml")
except Exception:
    SKRL_PPO_CFG_PATH = None
__all__ = ["HumanoidSNCPPORunnerCfg", "SKRL_PPO_CFG_PATH"]
