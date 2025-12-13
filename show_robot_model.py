#!/usr/bin/env python3

"""Show SNC R5 robot model without training - just for visualization."""

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

def show_robot_model():
    """Show robot model in standing pose."""
    
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
    print("SHOWING SNC R5 ROBOT MODEL")
    print("=" * 80)
    print("Press Ctrl+C to exit")
    print("Observe the robot pose and joint positions:")
    print("- Check left vs right leg symmetry")
    print("- Check hip-leg connections")
    print("- Check overall stance")
    print("=" * 80)
    
    # Get initial joint positions
    joint_names = robot.joint_names
    initial_pos = robot.joint_pos
    
    print(f"\nCurrent Joint Positions:")
    for i, name in enumerate(joint_names):
        if any(keyword in name.lower() for keyword in ['hip', 'knee', 'ankle']):
            print(f"{name:25}: {initial_pos[0, i]:8.3f}")
    
    # Keep robot in standing pose
    target_pos = robot.joint_pos.clone()
    
    try:
        step_count = 0
        while True:
            # Maintain standing pose
            robot.set_joint_position_target(target_pos)
            
            # Step simulation
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=sim.get_physics_dt())
            
            step_count += 1
            
            # Print pose info every 500 steps (~10 seconds)
            if step_count % 500 == 0:
                current_pos = robot.joint_pos
                base_pos = robot.root_pos_w
                base_ori = robot.root_quat_w
                
                print(f"\nStep {step_count}:")
                print(f"Base position: [{base_pos[0,0]:.3f}, {base_pos[0,1]:.3f}, {base_pos[0,2]:.3f}]")
                print(f"Robot standing at height: {base_pos[0,2]:.3f}m")
                
                # Check if robot is stable
                if base_pos[0,2] < 0.5:
                    print("⚠️  Robot is sinking or falling!")
                elif base_pos[0,2] > 1.5:
                    print("⚠️  Robot is floating!")
                else:
                    print("✅ Robot is stable")
            
            time.sleep(0.01)  # 100 Hz
            
    except KeyboardInterrupt:
        print("\nExiting visualization...")

def main():
    """Main function."""
    try:
        show_robot_model()
    finally:
        # close the app
        simulation_app.close()

if __name__ == "__main__":
    main()