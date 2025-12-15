#!/usr/bin/env python3

"""Debug script to test left vs right leg joint control for SNC R5."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher

# launch the app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG

def run_debug():
    """Test left vs right leg joint movements."""
    
    # Create simulation context
    sim = SimulationContext()
    
    # Create scene
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Add ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    
    # Add robot
    robot_cfg = HUMANOID_SNC_R5_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    
    # Design the scene
    scene["robot"] = robot
    
    # Play simulation
    sim.reset()
    
    print("=" * 80)
    print("DEBUGGING LEFT LEG STIFFNESS ISSUE")
    print("=" * 80)
    
    # Get joint names
    joint_names = robot.joint_names
    left_joints = [name for name in joint_names if "left" in name and ("knee" in name or "hip" in name or "ankle" in name)]
    right_joints = [name for name in joint_names if "right" in name and ("knee" in name or "hip" in name or "ankle" in name)]
    
    print(f"\nLeft leg joints: {left_joints}")
    print(f"Right leg joints: {right_joints}")
    
    # Check joint indices
    left_indices = [robot.joint_names.index(name) for name in left_joints]
    right_indices = [robot.joint_names.index(name) for name in right_joints]
    
    print(f"\nLeft leg joint indices: {left_indices}")
    print(f"Right leg joint indices: {right_indices}")
    
    # Test joint limits
    joint_limits = robot.joint_pos_limits
    print(f"\nJoint position limits shape: {joint_limits.shape}")
    
    for i, (left_idx, right_idx) in enumerate(zip(left_indices, right_indices)):
        left_name = left_joints[i]
        right_name = right_joints[i]
        left_limits = joint_limits[0, left_idx]
        right_limits = joint_limits[0, right_idx]
        
        print(f"{left_name:25} limits: [{left_limits[0]:8.3f}, {left_limits[1]:8.3f}]")
        print(f"{right_name:25} limits: [{right_limits[0]:8.3f}, {right_limits[1]:8.3f}]")
        print()
    
    # Test initial positions
    initial_pos = robot.joint_pos
    print(f"\nInitial joint positions:")
    
    for i, (left_idx, right_idx) in enumerate(zip(left_indices, right_indices)):
        left_name = left_joints[i]
        right_name = right_joints[i]
        left_pos = initial_pos[0, left_idx]
        right_pos = initial_pos[0, right_idx]
        
        print(f"{left_name:25} pos: {left_pos:8.3f}")
        print(f"{right_name:25} pos: {right_pos:8.3f}")
        print()
    
    # Test movement by applying different positions
    print("Testing joint movement - applying knee bend...")
    
    # Create target positions - bend knees
    target_pos = robot.joint_pos.clone()
    
    # Find knee joint indices
    left_knee_idx = robot.joint_names.index("left_knee_joint")
    right_knee_idx = robot.joint_names.index("right_knee_joint")
    
    # Bend knees significantly
    target_pos[0, left_knee_idx] = 1.0   # Bend left knee
    target_pos[0, right_knee_idx] = 1.0  # Bend right knee
    
    # Apply positions for several steps to see the movement
    for step in range(100):
        # Set joint targets
        robot.set_joint_position_target(target_pos)
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())
        
        if step % 20 == 0:
            current_pos = robot.joint_pos
            left_current = current_pos[0, left_knee_idx]
            right_current = current_pos[0, right_knee_idx]
            print(f"Step {step}: Left knee: {left_current:.3f}, Right knee: {right_current:.3f}")
    
    print("Debug complete. Check if both knees moved equally.")

def main():
    """Main function."""
    try:
        run_debug()
    finally:
        # close the app
        simulation_app.close()

if __name__ == "__main__":
    main()