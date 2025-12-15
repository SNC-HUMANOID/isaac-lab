#!/usr/bin/env python3

"""
Script to visualize SNC R5 robot and check left leg joints
"""

import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Check SNC R5 left leg joints")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG

def main():
    """Main function."""
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])
    
    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # Robot
    robot_cfg = HUMANOID_SNC_R5_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print(f"[INFO]: Spawned robot: {robot.cfg.prim_path}")
    print(f"[INFO]: Number of joints: {robot.num_joints}")
    
    # Check all joint names
    print("\n[INFO]: Joint Names:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i:2d}: {name}")
    
    # Check left leg joint positions specifically
    print(f"\n[INFO]: Joint positions at reset:")
    joint_pos = robot.data.joint_pos[0]  # First environment
    for i, name in enumerate(robot.joint_names):
        if "left" in name and ("hip" in name or "knee" in name or "ankle" in name):
            print(f"  {name}: {joint_pos[i]:.4f}")
    
    # Check joint limits for left leg
    print(f"\n[INFO]: Left leg joint limits:")
    for i, name in enumerate(robot.joint_names):
        if "left" in name and ("hip" in name or "knee" in name or "ankle" in name):
            print(f"  {name}: [{robot.data.soft_joint_pos_limits[0][i][0]:.3f}, {robot.data.soft_joint_pos_limits[0][i][1]:.3f}]")
    
    # Let's test knee joints specifically
    print(f"\n[INFO]: Testing knee joint movement...")
    
    # Simulate for a few steps with zero actions
    for count in range(100):
        # Set a simple knee bend action
        if count > 20 and count < 60:
            # Try to bend knees
            actions = torch.zeros(robot.num_joints)
            # Find knee joints
            for i, name in enumerate(robot.joint_names):
                if "knee" in name:
                    actions[i] = 0.3  # Try to bend knee
            
            robot.set_joint_position_target(actions.unsqueeze(0))
        
        # Perform step
        sim.step()
        
        # Get current joint positions every 20 steps
        if count % 20 == 0:
            joint_pos = robot.data.joint_pos[0]
            left_knee_idx = None
            right_knee_idx = None
            for i, name in enumerate(robot.joint_names):
                if name == "left_knee_joint":
                    left_knee_idx = i
                elif name == "right_knee_joint":
                    right_knee_idx = i
            
            if left_knee_idx is not None and right_knee_idx is not None:
                print(f"  Step {count:3d} - Left knee: {joint_pos[left_knee_idx]:.4f}, Right knee: {joint_pos[right_knee_idx]:.4f}")

if __name__ == "__main__":
    main()
    simulation_app.close()