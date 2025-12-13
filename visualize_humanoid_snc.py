#!/usr/bin/env python3

"""Script to visualize the Humanoid_SNC URDF in Isaac Sim"""

import argparse
import numpy as np
from isaaclab.app import AppLauncher

def main():
    """Main function to visualize the Humanoid_SNC robot"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize Humanoid_SNC robot in Isaac Sim")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    
    # Launch Isaac Sim
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    
    # Import after app launch
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    import isaaclab.sim as sim_utils
    
    @configclass
    class HumanoidSNCSceneCfg(InteractiveSceneCfg):
        """Configuration for the Humanoid SNC scene"""
        
        # Ground plane (remove for simplicity in initial visualization)
        
        # Robot configuration
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd",
                activate_contact_sensors=False,  # Disable for visualization
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),  # Start 1m above ground
                rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion
            ),
            actuators={},  # Empty actuators dict for visualization only
        )
    
    def run_simulator():
        """Run the interactive simulator"""
        
        # Create simulation context
        sim_cfg = sim_utils.SimulationCfg(dt=1/60.0)
        sim = SimulationContext(sim_cfg)
        
        # Set main camera
        sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 1.0])
        
        # Create scene
        scene_cfg = HumanoidSNCSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)
        
        # Play simulation
        sim.reset()
        
        print("\n" + "="*50)
        print("Humanoid_SNC Robot Visualization")
        print("="*50)
        print(f"Number of environments: {scene.num_envs}")
        print(f"Robot DoF: {scene['robot'].num_joints}")
        print(f"Joint names: {scene['robot'].joint_names}")
        print("="*50)
        print("Controls:")
        print("- Mouse: Orbit camera")
        print("- WASD: Move camera")
        print("- Q/E: Move camera up/down") 
        print("- Press ESC or close window to exit")
        print("="*50 + "\n")
        
        # Simulation loop
        while simulation_app.is_running():
            # Step simulation
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.cfg.dt)
    
    # Run the simulator
    run_simulator()
    
    # Close simulation
    simulation_app.close()

if __name__ == "__main__":
    main()