#!/usr/bin/env python3
"""Simple viewer for SNC R5 robot model in Isaac Sim."""

from isaaclab.app import AppLauncher

# Create launcher
app_launcher = AppLauncher(headless=False)  # GUI mode
simulation_app = app_launcher.app

import omni.isaac.core.utils.prims as prim_utils
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg

# Import robot config
from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG

def main():
    """Display SNC R5 robot model."""

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Setup scene
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 1.0])

    # Add ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Add lighting
    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)

    # Create robot
    robot_cfg = HUMANOID_SNC_R5_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=robot_cfg)

    # Reset simulation
    sim.reset()

    print("="*80)
    print("🤖 SNC R5 HUMANOID ROBOT VIEWER")
    print("="*80)
    print(f"Robot spawned at: {robot.cfg.prim_path}")
    print(f"Number of joints: {robot.num_joints}")
    print(f"Joint names: {robot.joint_names}")
    print("\n💡 Controls:")
    print("  - Mouse: Rotate camera")
    print("  - Scroll: Zoom in/out")
    print("  - WASD: Move camera")
    print("  - Press SPACE to start simulation")
    print("  - Press ESC or close window to exit")
    print("="*80)

    # Simulation loop
    while simulation_app.is_running():
        # Step simulation
        sim.step()

        # Update robot
        robot.update(dt=sim.get_physics_dt())

    # Cleanup
    simulation_app.close()

if __name__ == "__main__":
    main()
