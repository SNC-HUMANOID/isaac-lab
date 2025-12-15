#!/usr/bin/env python3
"""Force left leg to step forward - test if it CAN step."""

import torch
import gymnasium as gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

env = gym.make("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", num_envs=args.num_envs)
obs, _ = env.reset()

robot = env.unwrapped.scene["robot"]

# Find joint indices
joint_indices = {}
for i, name in enumerate(robot.data.joint_names):
    joint_indices[name] = i

print("="*80)
print("FORCING LEFT LEG TO STEP FORWARD")
print("="*80)

# Get left leg joints
left_hip_pitch = joint_indices.get("left_hip_pitch_joint")
left_knee = joint_indices.get("left_knee_joint")
left_ankle_pitch = joint_indices.get("left_ankle_pitch_joint")

print(f"\nLeft leg joints:")
print(f"  Hip pitch:   index {left_hip_pitch}")
print(f"  Knee:        index {left_knee}")
print(f"  Ankle pitch: index {left_ankle_pitch}")

print("\nTest 1: Move left hip forward (negative pitch)")
print("-" * 80)

for step in range(300):
    actions = torch.zeros((args.num_envs, env.action_space.shape[0]))

    # Phase 1: Lift left knee (0-100 steps)
    if step < 100:
        actions[:, left_knee] = 2.0  # Bend knee more
        actions[:, left_hip_pitch] = -1.0  # Hip forward

    # Phase 2: Swing forward (100-200 steps)
    elif step < 200:
        actions[:, left_hip_pitch] = -3.0  # Strong forward
        actions[:, left_knee] = 1.0  # Slightly bent

    # Phase 3: Plant foot (200-300 steps)
    else:
        actions[:, left_hip_pitch] = 0.0  # Return to center
        actions[:, left_knee] = 0.0
        actions[:, left_ankle_pitch] = 0.5  # Dorsiflexion

    obs, reward, terminated, truncated, info = env.step(actions)

    if step % 50 == 0:
        print(f"\nStep {step}:")
        hip_pos = robot.data.joint_pos[0, left_hip_pitch].item()
        knee_pos = robot.data.joint_pos[0, left_knee].item()
        ankle_pos = robot.data.joint_pos[0, left_ankle_pitch].item()

        base_pos = robot.data.root_pos_w[0].cpu().numpy()

        print(f"  Left hip pitch:   {hip_pos:6.3f} rad")
        print(f"  Left knee:        {knee_pos:6.3f} rad")
        print(f"  Left ankle pitch: {ankle_pos:6.3f} rad")
        print(f"  Base position:    x={base_pos[0]:.3f}, y={base_pos[1]:.3f}, z={base_pos[2]:.3f}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

final_hip = robot.data.joint_pos[0, left_hip_pitch].item()
init_hip = -0.20  # From init state

print(f"\nLeft hip pitch:")
print(f"  Initial:  {init_hip:.3f} rad")
print(f"  Final:    {final_hip:.3f} rad")
print(f"  Change:   {final_hip - init_hip:.3f} rad")

if abs(final_hip - init_hip) < 0.1:
    print("\n❌ LEFT HIP BARELY MOVED!")
    print("   Possible causes:")
    print("   1. Action scale too small (0.5 * action)")
    print("   2. Stiffness too high - resists movement")
    print("   3. Joint direction wrong")
else:
    print("\n✅ Left hip moved significantly")
    if final_hip < init_hip:
        print("   Hip moved FORWARD (more negative)")
    else:
        print("   Hip moved BACKWARD (more positive)")

env.close()
print("\n" + "="*80)
