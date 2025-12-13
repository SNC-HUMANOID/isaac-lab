# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
import torch
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic

# Safety patches for numerical stability
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
    std = torch.nan_to_num(std, nan=0.1, posinf=5.0, neginf=1e-4)  # ลด std เริ่มต้น
    std = torch.clamp(std, min=1e-5, max=5.0)  # ลด max std
    self.distribution = Normal(mean, std)

def _rebuild_safe_distribution(dist: Normal) -> Normal:
    loc = torch.nan_to_num(getattr(dist, "loc"), nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.nan_to_num(getattr(dist, "scale"), nan=0.1, posinf=5.0, neginf=1e-4)
    scale = torch.clamp(scale, min=1e-5, max=5.0)
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
    value = torch.clamp(value, min=-500.0, max=500.0)
    return value

# Apply patches
ActorCritic.update_distribution = patched_update_distribution
ActorCritic.act = patched_act
ActorCritic.evaluate = patched_evaluate
if hasattr(ActorCritic, "act_inference"):
    ActorCritic.act_inference = patched_act_inference

@configclass
class HumanoidSNCBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Optimized PPO configuration for balance training"""

    # === Training Parameters ===
    num_steps_per_env = 32  # ลดลงเพื่อ update บ่อยขึ้น
    max_iterations = 2000  # เพิ่มขึ้นเพื่อให้เรียนรู้ได้นานขึ้น
    save_interval = 100
    experiment_name = "humanoid_snc_balance_v2"
    empirical_normalization = True  # เปิดเพื่อ normalize observations

    # === Network Configuration ===
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,  # ลดลงเพื่อเริ่มต้นด้วยความระมัดระวัง
        actor_hidden_dims=[512, 256, 128],  # เพิ่มขนาด network
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # === Algorithm Configuration ===
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,  # เพิ่มขึ้น - value function สำคัญสำหรับ balance
        use_clipped_value_loss=True,
        clip_param=0.15,  # ลดลงเล็กน้อยเพื่อความระมัดระวัง
        entropy_coef=0.005,  # ลดลงเพื่อลด exploration ในช่วงแรก
        num_learning_epochs=8,  # เพิ่มขึ้นเพื่อเรียนรู้ดีขึ้น
        num_mini_batches=8,  # เพิ่มขึ้นเพื่อ gradient ที่เสถียร
        learning_rate=1e-4,  # ลดลงเพื่อความเสถียร
        schedule="adaptive",
        gamma=0.995,  # เพิ่มขึ้นเล็กน้อยเพื่อให้ long-term reward สำคัญขึ้น
        lam=0.98,  # เพิ่มขึ้นเพื่อ GAE ที่ดีขึ้น
        desired_kl=0.008,  # ลดลงเพื่อ policy update ที่ระมัดระวัง
        max_grad_norm=0.8,  # ลดลงเพื่อป้องกัน gradient explosion
    )

@configclass
class HumanoidSNCWalkingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for walking training (after balance is learned)"""

    # === Training Parameters ===
    num_steps_per_env = 64
    max_iterations = 3000
    save_interval = 150
    experiment_name = "humanoid_snc_walking_v2"
    empirical_normalization = True

    # === Network Configuration ===
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,  # ลดลงมากเพื่อ fine-tuning
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # === Algorithm Configuration ===
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.5,
        use_clipped_value_loss=True,
        clip_param=0.1,  # ลดลงมากสำหรับ fine-tuning
        entropy_coef=0.001,  # ลดลงมากเพื่อใช้ policy ที่เรียนรู้แล้ว
        num_learning_epochs=10,
        num_mini_batches=8,
        learning_rate=5e-5,  # ลดลงมากสำหรับ fine-tuning
        schedule="adaptive",
        gamma=0.998,
        lam=0.99,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )