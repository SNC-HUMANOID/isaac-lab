#!/usr/bin/env python3

"""Visualize Humanoid SNC robot with better initial pose."""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize Humanoid SNC")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class HumanoidSceneCfg(InteractiveSceneCfg):
    """Configuration for humanoid visualization scene."""
    
    # Robot with better initial pose
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
            copy_from_source=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.2),  # เพิ่มความสูง
            rot=(1.0, 0.0, 0.0, 0.0),  # ให้ยืนตรง
            # ใช้ตำแหน่งที่อยู่ในขอบเขต joint limits
            joint_pos={
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,  # ใช้ 0.0 แทน 0.1
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,  # ใช้ 0.0 แทน -0.2
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,  # ใช้ 0.0 แทน -0.1
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,  # ใช้ 0.0 แทน -0.2
                "Pelvis_joint": 0.0,
                "left_hip_pitch_joint": 0.0,  # ใช้ 0.0 แทน -0.1
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_ankle_pitch_joint": 0.0,  # ใช้ 0.0 แทน 0.1
                "left_ankle_roll_joint": 0.0,
                "left_ankle_yaw_joint": 0.0,
                "right_hip_pitch_joint": 0.0,  # ใช้ 0.0 แทน -0.1
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_ankle_pitch_joint": 0.0,  # ใช้ 0.0 แทน 0.1
                "right_ankle_roll_joint": 0.0,
                "right_ankle_yaw_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=400.0,  # เพิ่ม effort
                velocity_limit=10.0,
                stiffness=120.0,  # เพิ่ม stiffness
                damping=6.0,      # เพิ่ม damping
            ),
        },
    )


def main():
    """Main visualization function."""
    
    # Create ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # Add some lighting
    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
    cfg_light.func("/World/Light", cfg_light)
    
    # Initialize simulation context
    sim_cfg = SimulationCfg(dt=1/60)
    sim = SimulationContext(sim_cfg)
    
    # Create scene
    scene_cfg = HumanoidSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulation
    sim.reset()
    print("[INFO] Humanoid SNC visualization started")
    print("[INFO] Press Ctrl+C to stop")
    
    # Get robot reference
    robot: Articulation = scene["robot"]
    
    print(f"[INFO] Robot joint names: {robot.joint_names}")
    print(f"[INFO] Number of joints: {robot.num_joints}")
    print(f"[INFO] Robot height: {robot.data.root_pos_w[..., 2].mean():.3f}m")
    
    # Apply PD control to keep robot upright - ใช้ตำแหน่งที่ปลอดภัย
    target_positions = torch.zeros_like(robot.data.joint_pos)
    target_positions[:, :] = torch.tensor([
        0.0,   # left_shoulder_pitch_joint
        0.0,   # left_shoulder_roll_joint  
        0.0,   # left_shoulder_yaw_joint
        0.0,   # left_elbow_joint
        0.0,   # right_shoulder_pitch_joint
        0.0,   # right_shoulder_roll_joint
        0.0,   # right_shoulder_yaw_joint
        0.0,   # right_elbow_joint
        0.0,   # Pelvis_joint
        0.0,   # left_hip_pitch_joint
        0.0,   # left_hip_roll_joint
        0.0,   # left_hip_yaw_joint
        0.0,   # left_ankle_pitch_joint
        0.0,   # left_ankle_roll_joint
        0.0,   # left_ankle_yaw_joint
        0.0,   # right_hip_pitch_joint
        0.0,   # right_hip_roll_joint
        0.0,   # right_hip_yaw_joint
        0.0,   # right_ankle_pitch_joint
        0.0,   # right_ankle_roll_joint
        0.0,   # right_ankle_yaw_joint
    ], device=robot.device)
    
    # Run simulation
    count = 0
    while simulation_app.is_running():
        
        # Apply position control to keep robot standing
        robot.set_joint_position_target(target_positions)
        
        # Step physics
        scene.update(dt=sim_cfg.dt)
        
        if count % 60 == 0:  # Print every second
            height = robot.data.root_pos_w[..., 2].mean()
            print(f"[INFO] Step: {count}, Robot height: {height:.3f}m")
            
            # Reset if robot falls too low
            if height < 0.5:
                print("[INFO] Robot fell, resetting...")
                scene.reset()
        
        count += 1
    
    print("[INFO] Visualization ended")
    simulation_app.close()


if __name__ == "__main__":
    main()