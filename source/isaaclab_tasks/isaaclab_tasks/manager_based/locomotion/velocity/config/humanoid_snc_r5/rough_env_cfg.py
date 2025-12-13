# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Humanoid SNC R5 locomotion environment - based on G1."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

# Import custom action that inverts left hip
from .inverted_hip_action import InvertedLeftHipJointAction

##
# Pre-defined configs
##
from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG


@configclass
class HumanoidSNCR5Rewards(RewardsCfg):
    """Reward terms for Humanoid SNC R5 - based on G1 rewards."""

    # -- G1-EXACT reward structure (proven to work!)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)  # G1 exact
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.5,  # ลดลง - ให้โฟกัสที่การก้าวขามากกว่าความเร็ว
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}  # G1 exact
    )
    # Feet air time - Use simple version that doesn't require single stance
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,  # Simple version - rewards ANY foot lift
        weight=15.0,  # เพิ่มสูงมาก - บังคับให้ยกเท้า (แก้ปัญหาเดินช้า)
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_.*"),
            "threshold": 0.5,  # เพิ่มเวลายกเท้านานขึ้น
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,  # เพิ่มโทษสูงมาก - ห้ามลากเท้า บังคับให้ยกเท้า
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_.*"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,  # VERY LOW - allow hip movement for walking
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,  # Moderate penalty to allow some natural arm swing
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint", 
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_1_joint",
                    ".*_elbow_2_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,  # G1-style torso control - stable posture
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="Pelvis_joint")},
    )

    # เพิ่ม: ให้รางวัลเมื่อเข่างอขณะเดิน (ช่วยให้ยกขาสูงขึ้น)
    # Penalize knee being too straight during motion - encourage knee flexion
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,  # ลดโทษ - อนุญาตให้เข่างอได้มากขึ้น
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint")},
    )


@configclass 
class HumanoidSNCR5RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Humanoid SNC R5 locomotion environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene settings
        self.scene.robot = HUMANOID_SNC_R5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # แก้ไขพื้นลื่น - เพิ่ม friction ให้สูงขึ้น
        import isaaclab.sim as sim_utils
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,   # เพิ่มจาก 1.0 → 2.0 (แรงเสียดทานสูงขึ้น)
            dynamic_friction=1.5,  # เพิ่มจาก 1.0 → 1.5
            restitution=0.0,       # ไม่มีการตีกลับ
        )

        # scale down the terrains because the robot is smaller
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Use simple scale - wrapper will handle left hip inversion
        self.actions.joint_pos.scale = 0.5

        # ⚠️ TEMPORARY: Comment out arm exclusion to test old model
        # ล็อคแขนไม่ให้ขยับ - ตั้ง action scale = 0 สำหรับข้อต่อแขน
        # RL agent จะไม่ควบคุมแขน (แขนจะอยู่ในตำแหน่ง default)
        from isaaclab.managers import SceneEntityCfg
        self.actions.joint_pos.joint_names = [
            ".*_hip_.*",
            ".*_knee_joint",
            ".*_ankle_.*",
            "Pelvis_joint"
        ]  # ควบคุมเฉพาะขา + torso (ไม่รวมแขน)

        # rewards
        self.rewards: HumanoidSNCR5Rewards = HumanoidSNCR5Rewards()

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]

        # -- robot specific settings
        # update sensor period
        self.scene.contact_forces.update_period = self.decimation * self.sim.dt

        # randomization
        self.events.push_robot = None
        self.events.add_base_mass = None  # Disable problematic mass randomization
        self.events.base_com = None  # Disable center of mass randomization
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]

        # เพิ่ม friction ของเท้าหุ่นยนต์ (ไม่ให้ลื่น)
        self.events.physics_material.params["static_friction_range"] = (1.5, 2.0)  # เพิ่มจาก (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (1.2, 1.5)  # เพิ่มจาก (0.6, 0.6)

        # Adjusted rewards for walking
        self.rewards.lin_vel_z_l2.weight = 0.0  # G1 exact
        self.rewards.undesired_contacts = None  # Disable since no THIGH bodies
        self.rewards.flat_orientation_l2.weight = -0.2  # REDUCED - allow natural sway
        self.rewards.action_rate_l2.weight = -0.001  # REDUCED - allow bigger steps
        self.rewards.dof_acc_l2.weight = -1.25e-7  # G1 EXACT
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7  # G1 exact
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # Start with VERY SLOW walking - learn proper gait first!
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)  # ช้ามาก - เรียนรู้การก้าวขาก่อน
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)  # ลดการเดินข้าง
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)  # ลดการหมุน

        # terminations - G1 style
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class HumanoidSNCR5RoughEnvCfg_PLAY(HumanoidSNCR5RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None