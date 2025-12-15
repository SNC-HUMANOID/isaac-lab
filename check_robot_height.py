#!/usr/bin/env python3
"""Check the actual height of the humanoid robot."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from omni.isaac.core.simulation_context import SimulationContext
from isaaclab.assets import Articulation
from isaaclab_assets.robots.humanoid_10102025 import HUMANOID_10102025_CFG

# Create simulation
sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)

# Create robot
robot_cfg = HUMANOID_10102025_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

# Reset and step simulation
sim.reset()
for _ in range(10):
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    sim.step()

# Update robot data
robot.update(dt=0.01)

print("\n" + "="*80)
print("ROBOT INFORMATION:")
print("="*80)
print(f"Root position: {robot.data.root_pos_w[0].tolist()}")
print(f"Root height (Z): {robot.data.root_pos_w[0, 2].item():.4f} m")

# Get body positions
print(f"\nBody names and heights:")
for i, name in enumerate(robot.body_names):
    if 'ankle' in name.lower() or 'foot' in name.lower() or 'pelvis' in name.lower():
        pos = robot.data.body_pos_w[0, i]
        print(f"  {name:30s}: Z = {pos[2].item():7.4f} m")

print("="*80)

simulation_app.close()
