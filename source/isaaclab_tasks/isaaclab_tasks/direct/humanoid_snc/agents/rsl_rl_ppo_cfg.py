# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
import torch
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic

DEBUG_STD = False
DEBUG_VALUE = False

def patched_update_distribution(self: ActorCritic, observations: torch.Tensor):
    if not torch.isfinite(observations).all():
        observations = torch.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
    mean = self.actor(observations)
    if getattr(self, "noise_std_type", None) == "scalar" or not hasattr(self, "log_std"):
        if not hasattr(self, "std"):
            raise ValueError("ActorCritic lacks both log_std and std")
        std = self.std.expand_as(mean)
    else:
        std = torch.exp(self.log_std).expand_as(mean)
    std = torch.nan_to_num(std, nan=0.2, posinf=10.0, neginf=1e-4)
    std = torch.clamp(std, min=1e-6, max=10.0)
    self.distribution = Normal(mean, std)

def _rebuild_safe_distribution(dist: Normal) -> Normal:
    loc = torch.nan_to_num(getattr(dist, "loc"), nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.nan_to_num(getattr(dist, "scale"), nan=0.2, posinf=10.0, neginf=1e-4)
    scale = torch.clamp(scale, min=1e-6, max=10.0)
    return Normal(loc, scale)

def patched_act(self: ActorCritic, observations: torch.Tensor, **kwargs):
    self.update_distribution(observations)
    self.distribution = _rebuild_safe_distribution(self.distribution)
    return self.distribution.sample()

def patched_act_inference(self: ActorCritic, observations: torch.Tensor, **kwargs):
    self.update_distribution(observations)
    self.distribution = _rebuild_safe_distribution(self.distribution)
    return self.distribution.mean

def patched_evaluate(self: ActorCritic, critic_observations: torch.Tensor, **kwargs):
    if not torch.isfinite(critic_observations).all():
        critic_observations = torch.nan_to_num(critic_observations, nan=0.0, posinf=0.0, neginf=0.0)
    value = self.critic(critic_observations)
    if not torch.isfinite(value).all():
        value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
    value = torch.clamp(value, min=-1000.0, max=1000.0)
    return value

ActorCritic.update_distribution = patched_update_distribution
ActorCritic.act = patched_act
ActorCritic.evaluate = patched_evaluate
if hasattr(ActorCritic, "act_inference"):
    ActorCritic.act_inference = patched_act_inference

@configclass
class HumanoidSNCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 1500
    save_interval = 50
    experiment_name = "humanoid_snc_standing"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
