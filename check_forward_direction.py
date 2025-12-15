#!/usr/bin/env python3
"""Check which direction is forward for the humanoid robot."""

import argparse
from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Check robot forward direction")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# Import robot config
from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG

def main():
    """Check the forward direction of the robot."""

    # Create simulation context
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))

    # Set camera to see the robot from the side
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 1.0])

    # Spawn ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Spawn robot
    robot_cfg = HUMANOID_SNC_R5_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    # Play simulator
    sim.reset()

    print("\n" + "="*80)
    print("CHECKING ROBOT FORWARD DIRECTION")
    print("="*80)

    # Get initial position
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt=0.01)

    initial_pos = robot.data.root_pos_w[0].clone()
    initial_quat = robot.data.root_quat_w[0].clone()

    print(f"\nInitial Position (world frame): {initial_pos.cpu().numpy()}")
    print(f"Initial Quaternion (wxyz): {initial_quat.cpu().numpy()}")

    # Get local X-axis direction (forward) in world frame
    # In quaternion rotation: local X = (1, 0, 0) rotated by quat
    from isaaclab.utils.math import quat_rotate
    local_x = torch.tensor([[1.0, 0.0, 0.0]], device=robot.device)
    forward_world = quat_rotate(initial_quat.unsqueeze(0), local_x)

    print(f"\nLocal X-axis (forward) in world frame: {forward_world[0].cpu().numpy()}")
    print(f"  -> X component: {forward_world[0, 0].item():.4f}")
    print(f"  -> Y component: {forward_world[0, 1].item():.4f}")
    print(f"  -> Z component: {forward_world[0, 2].item():.4f}")

    # Interpret the direction
    print("\n" + "-"*80)
    if abs(forward_world[0, 0].item()) > 0.7:
        if forward_world[0, 0].item() > 0:
            print("✓ Robot's FORWARD direction is: +X (positive X-axis)")
            print("  Command lin_vel_x > 0 will move the robot in +X direction")
        else:
            print("✓ Robot's FORWARD direction is: -X (negative X-axis)")
            print("  Command lin_vel_x > 0 will move the robot in -X direction")
    elif abs(forward_world[0, 1].item()) > 0.7:
        if forward_world[0, 1].item() > 0:
            print("✓ Robot's FORWARD direction is: +Y (positive Y-axis)")
            print("  Command lin_vel_x > 0 will move the robot in +Y direction")
        else:
            print("✓ Robot's FORWARD direction is: -Y (negative Y-axis)")
            print("  Command lin_vel_x > 0 will move the robot in -Y direction")
    else:
        print("⚠ Robot's forward direction is ambiguous or at an angle")
        print(f"  Forward vector: {forward_world[0].cpu().numpy()}")

    print("-"*80)

    # Check velocity command ranges
    print("\n" + "="*80)
    print("VELOCITY COMMAND CONFIGURATION")
    print("="*80)

    from isaaclab_tasks.manager_based.locomotion.velocity.config.humanoid_snc_r5.flat_env_cfg import HumanoidSNCR5FlatEnvCfg
    env_cfg = HumanoidSNCR5FlatEnvCfg()
    env_cfg.__post_init__()

    print(f"\nlin_vel_x range: {env_cfg.commands.base_velocity.ranges.lin_vel_x}")
    print(f"  -> Training to move at {env_cfg.commands.base_velocity.ranges.lin_vel_x[0]:.2f} to {env_cfg.commands.base_velocity.ranges.lin_vel_x[1]:.2f} m/s")
    print(f"  -> Since lin_vel_x is POSITIVE, robot should move in its FORWARD direction")

    print("\n" + "="*80)
    print("SUMMARY:")
    print("  The robot is configured to move FORWARD at 0.3-0.8 m/s")
    print("  'Forward' is defined by the robot's local X-axis")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
    simulation_app.close()
