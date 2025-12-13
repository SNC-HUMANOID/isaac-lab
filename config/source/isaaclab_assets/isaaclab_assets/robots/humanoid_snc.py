# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Humanoid SNC robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

HUMANOID_SNC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R2/Humanoid_SNC.usd",
        copy_from_source=False,
        activate_contact_sensors=True,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "Pelvis_joint": 0.0,
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_ankle_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_yaw_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_ankle": 0.0,
            "right_ankle_roll_joint": 0.0,
            "right_ankle_yaw_joint": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_shoulder_pitch.*": 10.0,
                ".*_shoulder_roll.*": 10.0,
                ".*_shoulder_yaw.*": 10.0,
                ".*_elbow.*": 2.0,
                ".*Pelvis.*": 10.0,
                ".*_hip_pitch.*": 10.0,
                ".*_hip_roll.*": 20.0,
                ".*_hip_yaw.*": 10.0,
                ".*_ankle_pitch.*": 2.0,
                ".*_ankle_yaw.*": 2.0,
                "right_ankle": 5.0,
                "left_ankle_joint": 5.0,
            },
            damping={
                ".*_shoulder_pitch.*": 5.0,
                ".*_shoulder_roll.*": 5.0,
                ".*_shoulder_yaw.*": 5.0,
                ".*_elbow.*": 1.0,
                ".*Pelvis.*": 5.0,
                ".*_hip_pitch.*": 5.0,
                ".*_hip_roll.*": 5.0,
                ".*_hip_yaw.*": 5.0,
                ".*_ankle_pitch.*": 1.0,
                ".*_ankle_yaw.*": 1.0,
                "right_ankle": 0.1,
                "left_ankle_joint": 0.1,
            },
        ),
    },
)

"""Configuration for Humanoid SNC robot with rigid contact sensors."""
HUMANOID_SNC_CONTACT_CFG = HUMANOID_SNC_CFG.copy()
HUMANOID_SNC_CONTACT_CFG.spawn.activate_contact_sensors = True