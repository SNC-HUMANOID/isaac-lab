#!/usr/bin/env python3
"""Inspect body names in Humanoid 10102025 USD file."""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.core.simulation_context import SimulationContext
from isaaclab.assets import Articulation
from isaaclab_assets.robots.humanoid_10102025 import HUMANOID_10102025_CFG

# Create simulation context
sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)

# Create robot
robot_cfg = HUMANOID_10102025_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

# Reset to initialize
sim.reset()

print("\n" + "="*80)
print("BODY NAMES IN HUMANOID 10102025:")
print("="*80)
for i, body_name in enumerate(robot.body_names):
    print(f"  {i:2d}. {body_name}")

print("\n" + "="*80)
print("JOINT NAMES IN HUMANOID 10102025:")
print("="*80)
for i, joint_name in enumerate(robot.joint_names):
    print(f"  {i:2d}. {joint_name}")

print("\n" + "="*80)

simulation_app.close()
