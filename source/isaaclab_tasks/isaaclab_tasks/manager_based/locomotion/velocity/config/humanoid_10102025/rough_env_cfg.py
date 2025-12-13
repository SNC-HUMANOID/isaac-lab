# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Humanoid 10102025 locomotion environment - Based on G1."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.humanoid_10102025 import HUMANOID_10102025_CFG


@configclass
class Humanoid10102025Rewards(RewardsCfg):
    """Reward terms for Humanoid 10102025 - Exact copy from G1."""

    

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # Encourage staying upright and at proper height
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.5,  # เพิ่มจาก -0.5 เพื่อให้ควบคุมความสูงได้ดีขึ้น
        params={"target_height": 0.75, "asset_cfg": SceneEntityCfg("robot")},
    )
    
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,  # เพิ่มเพื่อให้เดินไปข้างหน้า ไม่แค่ยืนฉีกขา
        params={"command_name": "base_velocity", "std": 0.4},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=2.0,  # เพิ่มเพื่อให้ตอบสนองคำสั่งหมุน
        params={"command_name": "base_velocity", "std": 0.5}
    )
    # ใช้ฟังก์ชันใหม่ที่บังคับให้ทั้งสองเท้ายกสลับกัน
    feet_air_time = RewTerm(
        func=mdp.alternating_feet_air_time,
        weight=5.0,  # รีเวิร์ดสูงเพื่อบังคับให้ยกทั้งสองเท้า
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,  # ต้องยกเท้าอย่างน้อย 0.4 วินาที
        },
    )
    # รีเวิร์ดเมื่อมีเท้าเดียวบนพื้น (single stance) - บังคับให้สลับขา
    feet_balance = RewTerm(
        func=mdp.feet_balance_biped,
        weight=2.0,  # รีเวิร์ดเมื่อยืนขาเดียว (อีกขายก)
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    # รีเวิร์ดความสมมาตร - บังคับให้ซ้าย-ขวายกเท้าเท่ากัน
    feet_symmetry = RewTerm(
        func=mdp.feet_symmetry_biped,
        weight=1.5,  # รีเวิร์ดเมื่อเท้าทั้งสองยกเท่ากัน (ไม่เอียงข้าง)
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-3.0,  # ลงโทษหนักมากเพื่อไม่ให้ลากเท้า
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # NEW: Keep feet flat on ground - ankle pitch should stay near zero
    ankle_pitch_flat = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,  # ลดลงเพื่อให้ยกเท้าได้อิสระขึ้น
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])},
    )
    
    # ALTERNATIVE: หรือใช้ L2 penalty สำหรับความเบี่ยงเบนที่รุนแรง
    # ankle_pitch_flat_l2 = RewTerm(
    #     func=mdp.joint_deviation_l2,
    #     weight=-2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])},
    # )
    
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,  # เพิ่มจาก -1.0 เพื่อป้องกันข้อเท้าไปถึงขีดจำกัด
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    
    # อนุญาตให้สะโพกและเข่าเคลื่อนไหวได้อิสระ
    hip_movement = RewTerm(
        func=mdp.joint_vel_l1,
        weight=0.0,  # ไม่ลงโทษ ไม่รีเวิร์ด - ให้เคลื่อนไหวได้ตามที่จำเป็น
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])},
    )
    
    knee_movement = RewTerm(
        func=mdp.joint_vel_l1,
        weight=0.0,  # ไม่ลงโทษ ไม่รีเวิร์ด - ให้งอเข่าได้อิสระ
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )
    
    # Penalize deviation from default of the joints that are not essential for locomotion
    # Only penalize yaw and roll, allow pitch for leg lifting
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )
    # Strongly penalize hip_roll deviation to keep legs from spreading too wide
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,  # เพิ่มมากขึ้นเพื่อป้องกันขากางกว้าง
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow.*",
                ],
            )
        },
    )

    # ลบหรือแก้ไขส่วนนี้:
    joint_deviation_wrists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_joint"],  # ✅ ใช้ชื่อตรงๆ ไม่ใช่ regex
            )
        },
    )
    
    # Penalize wrist velocity to prevent fast movements
    wrist_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_joint"],  # ✅ ใช้ชื่อตรงๆ
            )
        },
    )

    # 1. ล็อค shoulder_roll ให้แน่น (ห้ามกางแขน)
    shoulder_roll_locked = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,  # น้ำหนักสูงมาก
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_roll_joint"],
            )
        },
    )
    
    # 2. ควบคุม shoulder_yaw เบาๆ
    shoulder_yaw_control = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_yaw_joint"],
            )
        },
    )
    
    # 3. อนุญาตให้ shoulder_pitch เคลื่อนไหวได้อิสระ
    shoulder_pitch_control = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,  # น้ำหนักต่ำมาก หรือไม่ใส่ก็ได้
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_pitch_joint"],
            )
        },
    )
    
    # 4. ควบคุม elbow
    elbow_control = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_elbow.*"],
            )
        },
    )


@configclass
class Humanoid10102025RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Humanoid10102025Rewards = Humanoid10102025Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = HUMANOID_10102025_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.lin_vel_z_l2.weight = -1.0  # ลดจาก -2.0 เพื่อให้ยกเท้าได้ง่ายขึ้น
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.0  # ลดจาก -5.0 เพื่อให้โยกตัวเดินได้ง่ายขึ้น
        self.rewards.action_rate_l2.weight = -0.005  # เพิ่มกลับเป็นค่าเดิมเพื่อให้การเคลื่อนไหวนุ่มนวลขึ้น
        self.rewards.dof_acc_l2.weight = -1.25e-7  # เพิ่มกลับเป็นค่าเดิมเพื่อควบคุมความเร่งให้ดีขึ้น
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_knee_joint"]  # เพิ่ม knee กลับมาเพื่อควบคุมความเร่ง
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7  # เพิ่มกลับเป็นค่าเดิมเพื่อควบคุม torque ให้ดีขึ้น
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_ankle.*", ".*_knee_joint"]  # เพิ่ม knee กลับมา
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class Humanoid10102025RoughEnvCfg_PLAY(Humanoid10102025RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
