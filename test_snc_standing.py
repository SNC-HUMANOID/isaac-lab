#!/usr/bin/env python3
"""Simple test - can SNC R5 even stand?"""

import torch
import gymnasium as gym

# Create environment with just 1 robot
env = gym.make("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", num_envs=1, render_mode="rgb_array")
obs, _ = env.reset()

print("="*80)
print("TESTING: Can SNC R5 stand without falling?")
print("="*80)

robot = env.unwrapped.scene["robot"]

# Get initial pose
print("\nInitial joint positions:")
for i, name in enumerate(robot.data.joint_names):
    if "leg" in name or "knee" in name or "ankle" in name or "hip" in name:
        pos = robot.data.default_joint_pos[0, i].item()
        print(f"  {name:35s}: {pos:6.3f}")

print("\nTesting standing for 500 steps with ZERO actions (just hold default pose)...")

for step in range(500):
    # Zero actions = hold default position
    actions = torch.zeros((1, env.action_space.shape[0]))
    obs, reward, terminated, truncated, info = env.step(actions)

    if step % 100 == 0:
        base_pos = robot.data.root_pos_w[0]
        base_quat = robot.data.root_quat_w[0]
        print(f"\nStep {step}:")
        print(f"  Base height: {base_pos[2].item():.3f} m")
        print(f"  Base orientation (wxyz): [{base_quat[0].item():.3f}, {base_quat[1].item():.3f}, {base_quat[2].item():.3f}, {base_quat[3].item():.3f}]")
        print(f"  Terminated: {terminated[0].item()}")
        print(f"  Total reward: {reward[0].item():.3f}")

        # Check individual rewards
        if 'episode_rewards' in info:
            print("  Individual rewards:")
            for key in ['feet_air_time', 'flat_orientation_l2', 'termination_penalty']:
                if key in info['episode_rewards']:
                    val = info['episode_rewards'][key][0].item()
                    print(f"    {key:30s}: {val:7.4f}")

    if terminated[0]:
        print(f"\n❌ FAILED at step {step}: Robot fell/terminated")
        print("Termination reason:", info.get('terminations', 'unknown'))
        break
else:
    print(f"\n✅ SUCCESS: Robot stood for 500 steps!")

env.close()
