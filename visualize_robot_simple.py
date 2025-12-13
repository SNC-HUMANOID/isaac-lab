#!/usr/bin/env python3
"""Simple visualization to check robot stance."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.assets import Articulation
from isaaclab_assets.robots.humanoid_10102025 import HUMANOID_10102025_CFG
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for test scene."""
    robot: ArticulationCfg = HUMANOID_10102025_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# Create scene
scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
scene = InteractiveScene(scene_cfg)

print("\n" + "="*80)
print("INITIAL ROBOT STATE")
print("="*80)
print(f"Spawn height: {HUMANOID_10102025_CFG.init_state.pos[2]} m")

# Reset scene
scene.reset()

# Step simulation a few times
for i in range(50):
    scene.write_data_to_sim()
    simulation_app.update()

# Get robot data
robot = scene["robot"]
print(f"\nAfter {i+1} steps:")
print(f"Root position: {robot.data.root_pos_w[0].tolist()}")
print(f"Root height: {robot.data.root_pos_w[0, 2].item():.4f} m")

# Check feet positions
print(f"\nFeet positions:")
for i, name in enumerate(robot.body_names):
    if 'ankle' in name.lower():
        pos = robot.data.body_pos_w[0, i]
        print(f"  {name:30s}: Z = {pos[2].item():7.4f} m")

print("="*80)
print("\nPress Ctrl+C to exit...")

# Keep simulation running
while simulation_app.is_running():
    scene.write_data_to_sim()
    simulation_app.update()

simulation_app.close()
