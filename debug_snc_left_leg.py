#!/usr/bin/env python3
"""Debug SNC R5 left leg - check if it's actually moving."""

import torch
import gymnasium as gym
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Humanoid-SNC-R5-v0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def main():
    # Create environment
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    env.reset()

    print("\n" + "="*80)
    print("SNC R5 LEFT LEG DEBUG")
    print("="*80)

    # Get robot info
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

    print(f"\nRobot joints ({len(robot.data.joint_names)}):")
    left_leg_joints = []
    right_leg_joints = []

    for i, name in enumerate(robot.data.joint_names):
        if "left" in name and ("hip" in name or "knee" in name or "ankle" in name):
            left_leg_joints.append((i, name))
            print(f"  LEFT  {i:2d}: {name}")
        elif "right" in name and ("hip" in name or "knee" in name or "ankle" in name):
            right_leg_joints.append((i, name))
            print(f"  RIGHT {i:2d}: {name}")

    print(f"\nContact sensor bodies:")
    for i, name in enumerate(contact_sensor.data.body_names):
        if "ankle" in name or "foot" in name:
            print(f"  {i}: {name}")

    print("\n" + "="*80)
    print("RUNNING TEST - RANDOM ACTIONS")
    print("="*80)

    for step in range(300):
        # Random actions
        actions = torch.randn_like(env.action_space.sample())

        # Force left knee to move
        if step % 50 == 0:
            for idx, name in left_leg_joints:
                if "knee" in name:
                    actions[:, idx] = 1.0  # Max action on left knee

        obs, reward, terminated, truncated, info = env.step(actions)

        # Print every 50 steps
        if step % 50 == 0:
            print(f"\n--- Step {step} ---")

            # Check joint positions
            print("Joint positions (env 0):")
            for idx, name in left_leg_joints:
                pos = robot.data.joint_pos[0, idx].item()
                vel = robot.data.joint_vel[0, idx].item()
                print(f"  LEFT  {name:30s}: pos={pos:6.3f}, vel={vel:6.3f}")

            for idx, name in right_leg_joints:
                pos = robot.data.joint_pos[0, idx].item()
                vel = robot.data.joint_vel[0, idx].item()
                print(f"  RIGHT {name:30s}: pos={pos:6.3f}, vel={vel:6.3f}")

            # Check contact forces
            print("\nContact forces (env 0):")
            for i, name in enumerate(contact_sensor.data.body_names):
                if "ankle" in name:
                    force = contact_sensor.data.net_forces_w[0, i].norm().item()
                    air_time = contact_sensor.data.current_air_time[0, i].item()
                    contact_time = contact_sensor.data.current_contact_time[0, i].item()
                    status = "GROUND" if contact_time > 0 else "AIR"
                    print(f"  {name:30s}: force={force:6.1f}N  air={air_time:.2f}s  contact={contact_time:.2f}s [{status}]")

            # Check feet_air_time reward
            print(f"\nRewards (env 0):")
            print(f"  feet_air_time reward: {info['episode_rewards']['feet_air_time'][0].item():.4f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nCheck above output:")
    print("1. Do LEFT leg joints actually move? (pos and vel should change)")
    print("2. Does left ankle leave the ground? (air_time should increase)")
    print("3. Is feet_air_time reward positive?")
    print("\nIf LEFT leg doesn't move → Check actuator or URDF")
    print("If LEFT leg moves but no air_time → Check contact sensor")
    print("If air_time exists but reward negative → Check reward function")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
