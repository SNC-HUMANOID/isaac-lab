# SPDX-License-Identifier: BSD-3-Clause
"""
G1-style PPO configuration for Humanoid SNC
Uses proven G1 hyperparameters adapted for SNC robot
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
import torch
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic

# G1-style safety patches (proven in production)
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
    # G1-style std handling
    std = torch.nan_to_num(std, nan=0.8, posinf=2.0, neginf=1e-4)
    std = torch.clamp(std, min=1e-4, max=2.0)
    self.distribution = Normal(mean, std)

def _rebuild_safe_distribution(dist: Normal) -> Normal:
    loc = torch.nan_to_num(getattr(dist, "loc"), nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.nan_to_num(getattr(dist, "scale"), nan=0.8, posinf=2.0, neginf=1e-4)
    scale = torch.clamp(scale, min=1e-4, max=2.0)
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

# Apply patches
ActorCritic.update_distribution = patched_update_distribution
ActorCritic.act = patched_act
ActorCritic.evaluate = patched_evaluate
if hasattr(ActorCritic, "act_inference"):
    ActorCritic.act_inference = patched_act_inference

@configclass
class HumanoidSNCG1StylePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """G1-style PPO configuration adapted for Humanoid SNC"""

    # === Training Parameters (G1-proven) ===
    num_steps_per_env = 24  # G1 uses 24
    max_iterations = 2000   # Longer training for stability
    save_interval = 100
    experiment_name = "humanoid_snc_g1_style"
    empirical_normalization = False  # G1 uses False

    # === Network Configuration (G1-style) ===
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # G1 uses 1.0 (high exploration)
        actor_hidden_dims=[512, 256, 128],  # G1 rough terrain size
        critic_hidden_dims=[512, 256, 128],
        activation="elu",  # G1 uses ELU
    )

    # === Algorithm Configuration (G1-proven) ===
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # G1 uses 1.0
        use_clipped_value_loss=True,
        clip_param=0.2,  # G1 uses 0.2
        entropy_coef=0.008,  # G1 uses 0.008 (low entropy)
        num_learning_epochs=5,  # G1 uses 5
        num_mini_batches=4,  # G1 uses 4
        learning_rate=1.0e-3,  # G1 uses 1e-3 (higher than our previous)
        schedule="adaptive",  # G1 uses adaptive
        gamma=0.99,  # G1 uses 0.99
        lam=0.95,  # G1 uses 0.95
        desired_kl=0.01,  # G1 uses 0.01
        max_grad_norm=1.0,  # G1 uses 1.0
    )

@configclass
class HumanoidSNCG1FlatStylePPORunnerCfg(HumanoidSNCG1StylePPORunnerCfg):
    """G1 flat terrain style for faster training on flat ground"""

    def __post_init__(self):
        super().__post_init__()

        # Flat terrain optimizations (like G1 flat)
        self.max_iterations = 1500  # G1 flat uses 1500
        self.experiment_name = "humanoid_snc_g1_flat_style"

        # Smaller networks for flat terrain (G1 flat style)
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]

@configclass
class HumanoidSNCG1AdvancedPPORunnerCfg(HumanoidSNCG1StylePPORunnerCfg):
    """Advanced configuration with higher performance tweaks"""

    def __post_init__(self):
        super().__post_init__()

        # Advanced optimizations
        self.experiment_name = "humanoid_snc_g1_advanced"
        self.num_steps_per_env = 32  # More data per update
        self.max_iterations = 3000   # Extended training

        # Larger networks for complex behaviors
        self.policy.actor_hidden_dims = [512, 512, 256, 128]
        self.policy.critic_hidden_dims = [512, 512, 256, 128]

        # Fine-tuned hyperparameters
        self.algorithm.num_learning_epochs = 8  # More learning per batch
        self.algorithm.num_mini_batches = 8     # Smaller mini-batches
        self.algorithm.learning_rate = 8e-4     # Slightly lower LR
        self.algorithm.entropy_coef = 0.005     # Even lower entropy