# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat terrain locomotion config for Humanoid SNC R5."""

from isaaclab.utils import configclass
from .rough_env_cfg import HumanoidSNCR5RoughEnvCfg


@configclass
class HumanoidSNCR5FlatEnvCfg(HumanoidSNCR5RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # -------------------------------
        # ✅ Rewards
        # -------------------------------

        # Orientation penalty (เพิ่มขึ้นเพื่อป้องกันการนอน)
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = -2.0  # เพิ่มจาก -0.2 → -2.0  

        # Feet air-time (กระตุ้นให้ยกขา)
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.weight = 5.0  # สูงมาก - ให้รางวัลการยกเท้า
            self.rewards.feet_air_time.params["threshold"] = 0.2  # เพิ่มเวลายกเท้า  

        # Forward progress (เดินไปข้างหน้า)
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 5.0

        # ลดโทษแขน
        if hasattr(self.rewards, "joint_deviation_arms"):
            self.rewards.joint_deviation_arms.weight = -0.05
            
        # ลด termination penalty
        if hasattr(self.rewards, "termination_penalty"):
            self.rewards.termination_penalty.weight = -0.05

        # -------------------------------
        # ✅ Terminations
        # -------------------------------
        # ถ้ามี base_height termination ให้ตั้ง threshold ต่ำ = ถือว่าล้ม
        if hasattr(self.terminations, "base_height"):
            self.terminations.base_height.params["threshold"] = 0.7

        # -------------------------------
        # ✅ Scene: Flat terrain
        # -------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class HumanoidSNCR5FlatEnvCfg_PLAY(HumanoidSNCR5FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # จำนวน env สำหรับ mode เล่น/ทดสอบ
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # ปิด corruption และ pushing events
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
