#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize the Humanoid 10102025 robot model with IMU."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="View Humanoid 10102025 robot model.")
parser.add_argument("--with_imu", action="store_true", default=False, help="Load model with IMU sensor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

##
# Scene configuration
##

@configclass
class ViewerSceneCfg(InteractiveSceneCfg):
    """Configuration for the viewer scene."""

    # ground plane
    ground = sim_utils.GroundPlaneCfg()

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",  # Will be set dynamically
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={
                # Legs - slightly bent stance
                "left_hip_pitch_joint": -0.1,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.2,
                "left_ankle_pitch_joint": -0.1,
                "left_ankle_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.1,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_joint": 0.2,
                "right_ankle_pitch_joint": -0.1,
                "right_ankle_roll_joint": 0.0,
                # Arms - neutral position
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
            },
        ),
        actuators={
            "legs": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                stiffness=25.0,
                damping=5.0,
            ),
            "arms": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
                stiffness=15.0,
                damping=3.0,
            ),
        },
    )

    # lights
    dome_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))


##
# Main function
##

def main():
    """Main function to visualize the Humanoid 10102025 robot."""

    # Use the converted USD file
    usd_path = "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025.usd"
    print("[INFO] Loading Humanoid 10102025 model from converted USD")

    # Create simulation configuration
    sim_cfg = SimulationCfg(dt=0.01, device="cuda:0" if torch.cuda.is_available() else "cpu")
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set up the scene
    scene_cfg = ViewerSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot.spawn.usd_path = usd_path
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    print("[INFO] Setup complete. Robot model loaded successfully!")
    print("[INFO] Robot information:")
    print(f"  - Number of joints: {scene['robot'].num_joints}")
    print(f"  - Number of bodies: {scene['robot'].num_bodies}")
    print(f"  - Joint names: {scene['robot'].joint_names}")
    print(f"  - Body names: {scene['robot'].body_names}")

    if args_cli.with_imu and "imu_link" in scene['robot'].body_names:
        print("[INFO] ✓ IMU sensor detected in model!")

    print("\n[INFO] Use the viewer to inspect the robot model.")
    print("[INFO] Press Ctrl+C to exit.")

    # Simulation loop
    count = 0
    while simulation_app.is_running():
        # Perform a simulation step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

        # Print status every 100 steps
        count += 1
        if count % 100 == 0:
            print(f"[INFO] Simulation running... (step {count})")

    # Close the simulation
    print("[INFO] Closing simulation...")


if __name__ == "__main__":
    # Run main function
    main()
    # Close the simulation app
    simulation_app.close()
