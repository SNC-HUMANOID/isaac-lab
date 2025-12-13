#!/usr/bin/env python3

"""Test script to compare left vs right knee bending for SNC R5."""

import torch
import time
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

def test_knee_bending():
    """Test left vs right knee joint bending."""
    
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
    print("TESTING KNEE BENDING - LEFT VS RIGHT")
    print("=" * 80)
    
    # Get joint indices
    joint_names = robot.joint_names
    left_knee_idx = robot.joint_names.index("left_knee_joint")
    right_knee_idx = robot.joint_names.index("right_knee_joint")
    
    print(f"Left knee joint index: {left_knee_idx}")
    print(f"Right knee joint index: {right_knee_idx}")
    
    # Get joint limits
    joint_limits = robot.joint_pos_limits
    left_knee_limits = joint_limits[0, left_knee_idx]
    right_knee_limits = joint_limits[0, right_knee_idx]
    
    print(f"\nLeft knee limits: [{left_knee_limits[0]:.3f}, {left_knee_limits[1]:.3f}]")
    print(f"Right knee limits: [{right_knee_limits[0]:.3f}, {right_knee_limits[1]:.3f}]")
    
    # Test initial position
    initial_pos = robot.joint_pos
    print(f"\nInitial positions:")
    print(f"Left knee: {initial_pos[0, left_knee_idx]:.3f}")
    print(f"Right knee: {initial_pos[0, right_knee_idx]:.3f}")
    
    # Test sequence: straight -> bent -> straight
    test_positions = [
        (0.0, 0.0, "Both straight"),
        (0.5, 0.5, "Both bent 0.5"),
        (1.0, 1.0, "Both bent 1.0"), 
        (1.5, 1.5, "Both bent 1.5"),
        (2.0, 2.0, "Both max bent"),
        (1.0, 1.0, "Both bent 1.0"),
        (0.0, 0.0, "Both straight"),
        (1.0, 0.0, "Left bent, Right straight"),
        (0.0, 1.0, "Left straight, Right bent"),
        (0.3, 0.1, "Back to initial")
    ]
    
    for left_pos, right_pos, description in test_positions:
        print(f"\n--- {description} ---")
        print(f"Target: Left={left_pos:.1f}, Right={right_pos:.1f}")
        
        # Set target positions
        target_pos = robot.joint_pos.clone()
        target_pos[0, left_knee_idx] = left_pos
        target_pos[0, right_knee_idx] = right_pos
        
        # Apply for multiple steps to ensure movement
        for step in range(50):
            robot.set_joint_position_target(target_pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=sim.get_physics_dt())
        
        # Check final positions
        current_pos = robot.joint_pos
        left_actual = current_pos[0, left_knee_idx]
        right_actual = current_pos[0, right_knee_idx]
        
        print(f"Actual: Left={left_actual:.3f}, Right={right_actual:.3f}")
        print(f"Error: Left={abs(left_pos - left_actual):.3f}, Right={abs(right_pos - right_actual):.3f}")
        
        # Wait a bit
        time.sleep(0.5)
    
    print("\n" + "=" * 80)
    print("KNEE BENDING TEST COMPLETE")
    print("Check if both knees move equally and reach target positions")
    print("=" * 80)

def main():
    """Main function."""
    try:
        test_knee_bending()
        # Keep simulation running for observation
        print("\nPress Ctrl+C to exit...")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # close the app
        simulation_app.close()

if __name__ == "__main__":
    main()