# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable Actor-Critic implementation that ensures std > 0"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation

# Register the class in the global namespace so RSL-RL can find it
import sys
current_module = sys.modules[__name__]


class StableActorCritic(ActorCritic):
    """Actor-critic that enforces positive standard deviation"""

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,  
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        min_std=1e-4,
        **kwargs,
    ):
        # Store min_std before calling parent init
        self.min_std = min_std
        
        # Call parent constructor
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation with clamping
        if self.noise_std_type == "scalar":
            std = torch.clamp(self.std.expand_as(mean), min=self.min_std)
        elif self.noise_std_type == "log":
            std = torch.clamp(torch.exp(self.log_std).expand_as(mean), min=self.min_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)