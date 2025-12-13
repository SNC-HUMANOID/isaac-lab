# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Humanoid SNC R5 robot with tuned joints for locomotion."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

HUMANOID_SNC_R5_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees_relative.urdf",
        activate_contact_sensors=True,
        fix_base=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=200.0,
                damping=10.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),  # Base height
        joint_pos={
            # Legs - Adjusted for more upright standing pose
            "left_hip_pitch_joint": -0.10,   # Less forward lean (more upright)
            "right_hip_pitch_joint": -0.10,  # Less forward lean (more upright)
            ".*_knee_joint": 0.15,           # Less bent knees (more straight)
            ".*_ankle_pitch_joint": -0.05,   # Less ankle bend (more neutral)

            # Arms - งอตั้งฉาก (90 องศา)
            "left_shoulder_pitch_joint": 0.0,     # ไหล่ตรง
            "left_shoulder_roll_joint": 0.0,      # ไม่เอียง
            "right_shoulder_pitch_joint": 0.0,    # ไหล่ตรง
            "right_shoulder_roll_joint": 0.0,     # ไม่เอียง
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_1_joint": -1.57,            # งอ 90° (π/2 rad)
            ".*_elbow_2_joint": -1.57,            # งอ 90° (π/2 rad)

            # Torso
            "Pelvis_joint": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "Pelvis_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0,     # เพิ่มสำหรับ 45kg
                ".*_hip_roll_joint": 200.0,    # เพิ่มสำหรับ 45kg
                ".*_hip_pitch_joint": 250.0,   # เพิ่มสำหรับ 45kg (G1=200 × 1.25)
                ".*_knee_joint": 250.0,        # เพิ่มสำหรับ 45kg
                "Pelvis_joint": 250.0,         # เพิ่มสำหรับ 45kg
            },
            damping={
                ".*_hip_yaw_joint": 7.0,       # เพิ่มตามน้ำหนัก
                ".*_hip_roll_joint": 7.0,
                ".*_hip_pitch_joint": 7.0,
                ".*_knee_joint": 7.0,
                "Pelvis_joint": 7.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "Pelvis_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_1_joint",
                ".*_elbow_2_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=1000.0,   # ลด stiffness แขน
            damping=50.0,
            armature=0.01,
        ),
    },
)

"""Configuration with contact sensors enabled."""
HUMANOID_SNC_R5_CONTACT_CFG = HUMANOID_SNC_R5_CFG.copy()
HUMANOID_SNC_R5_CONTACT_CFG.spawn.activate_contact_sensors = True
