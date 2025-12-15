#!/usr/bin/env python3
"""Debug script to check H1 left leg joints and movements."""

import torch
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug H1 left leg")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import H1_MINIMAL_CFG

def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 1.0])

    # Spawn H1 robot
    robot_cfg = H1_MINIMAL_CFG.replace(prim_path="/World/envs/env_0/Robot")
    robot = Articulation(robot_cfg)

    # Design scene
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Play simulator
    sim.reset()
    robot.reset()

    print("\n" + "="*80)
    print("H1 ROBOT LEFT LEG DEBUG")
    print("="*80)

    # Get all joint names
    joint_names = robot.data.joint_names
    print("\nAll joints:")
    for i, name in enumerate(joint_names):
        print(f"  {i}: {name}")

    # Find left leg joints
    left_leg_indices = []
    left_leg_names = []
    for i, name in enumerate(joint_names):
        if "left" in name.lower() and ("hip" in name or "knee" in name or "ankle" in name):
            left_leg_indices.append(i)
            left_leg_names.append(name)

    print(f"\nLeft leg joints found: {len(left_leg_indices)}")
    for i, (idx, name) in enumerate(zip(left_leg_indices, left_leg_names)):
        print(f"  {i}: Joint {idx} = {name}")

    if len(left_leg_indices) == 0:
        print("\n❌ ERROR: No left leg joints found!")
        return

    # Find right leg joints for comparison
    right_leg_indices = []
    right_leg_names = []
    for i, name in enumerate(joint_names):
        if "right" in name.lower() and ("hip" in name or "knee" in name or "ankle" in name):
            right_leg_indices.append(i)
            right_leg_names.append(name)

    print(f"\nRight leg joints found: {len(right_leg_indices)}")
    for i, (idx, name) in enumerate(zip(right_leg_indices, right_leg_names)):
        print(f"  {i}: Joint {idx} = {name}")

    # Run simulation and test joints
    print("\n" + "="*80)
    print("TESTING LEFT LEG MOVEMENT")
    print("="*80)

    # Reset to default pose
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)

    for i in range(200):
        # Apply sinusoidal movement to left leg
        if i % 20 == 0:
            phase = i * 0.1
            test_pos = robot.data.default_joint_pos.clone()

            # Move left knee
            if len(left_leg_indices) > 0:
                for idx in left_leg_indices:
                    if "knee" in joint_names[idx]:
                        test_pos[0, idx] = robot.data.default_joint_pos[0, idx] + 0.3 * torch.sin(phase)
                        print(f"\nStep {i}: Moving {joint_names[idx]} to {test_pos[0, idx].item():.3f}")

            robot.set_joint_position_target(test_pos)

        # Step simulation
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        # Check actual positions every 50 steps
        if i % 50 == 0 and i > 0:
            print(f"\nStep {i} - Joint positions:")
            for idx, name in zip(left_leg_indices, left_leg_names):
                pos = robot.data.joint_pos[0, idx].item()
                vel = robot.data.joint_vel[0, idx].item()
                print(f"  {name}: pos={pos:.3f}, vel={vel:.3f}")

    print("\n" + "="*80)
    print("FINAL CHECK - Comparing Left vs Right")
    print("="*80)

    print("\nLeft leg final positions:")
    for idx, name in zip(left_leg_indices, left_leg_names):
        pos = robot.data.joint_pos[0, idx].item()
        print(f"  {name}: {pos:.3f}")

    print("\nRight leg final positions:")
    for idx, name in zip(right_leg_indices, right_leg_names):
        pos = robot.data.joint_pos[0, idx].item()
        print(f"  {name}: {pos:.3f}")

    print("\n✅ Debug complete. Check if left leg moved in the GUI.")

if __name__ == "__main__":
    main()
    simulation_app.close()
