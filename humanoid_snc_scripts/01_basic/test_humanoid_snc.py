#!/usr/bin/env python3

"""Simple test for Humanoid SNC robot."""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Test Humanoid SNC robot")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for test scene."""
    
    # Robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
            copy_from_source=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={
                ".*": 0.0,
            },
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=300.0,
                velocity_limit=10.0,
                stiffness=80.0,
                damping=4.0,
            ),
        },
    )

    # Ground plane - will be created in main function
    pass


def main():
    """Main test function."""
    
    # Create ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # Initialize simulation context
    sim_cfg = SimulationCfg(dt=1/60)
    sim = SimulationContext(sim_cfg)
    
    # Create scene
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulation
    sim.reset()
    print("[INFO] Simulation started. Robot loaded successfully!")
    
    # Get robot reference
    robot: Articulation = scene["robot"]
    
    print(f"[INFO] Robot joint names: {robot.joint_names}")
    print(f"[INFO] Number of joints: {robot.num_joints}")
    
    # Run for a few steps
    for count in range(1000):
        # Apply small random joint torques
        joint_efforts = torch.randn_like(robot.data.joint_pos) * 0.1
        robot.set_joint_effort_target(joint_efforts)
        
        # Step physics
        scene.update(dt=sim_cfg.dt)
        
        if count % 100 == 0:
            print(f"[INFO] Step: {count}")
    
    print("[INFO] Test completed successfully!")
    
    # Close simulation
    simulation_app.close()


if __name__ == "__main__":
    main()