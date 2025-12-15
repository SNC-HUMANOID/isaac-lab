#!/usr/bin/env python3
"""Force left leg to move - test if it CAN move at all."""

import torch
import gymnasium as gym

env = gym.make("Isaac-Velocity-Flat-Humanoid-SNC-R5-v0", num_envs=4)
obs, _ = env.reset()

robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

# Find left leg joint indices
left_leg_indices = {}
for i, name in enumerate(robot.data.joint_names):
    if "left" in name and ("hip" in name or "knee" in name or "ankle" in name):
        left_leg_indices[name] = i

print("="*80)
print("FORCING LEFT LEG TO MOVE")
print("="*80)
print(f"\nLeft leg joints: {list(left_leg_indices.keys())}")

for step in range(300):
    # Create action
    actions = torch.zeros((4, env.action_space.shape[0]))

    # Force left knee to move UP (bend more)
    if "left_knee_joint" in left_leg_indices:
        knee_idx = left_leg_indices["left_knee_joint"]
        # Sinusoidal movement
        actions[:, knee_idx] = 2.0 * torch.sin(step * 0.1)  # Strong movement

    # Force left hip to lift
    if "left_hip_pitch_joint" in left_leg_indices:
        hip_idx = left_leg_indices["left_hip_pitch_joint"]
        actions[:, hip_idx] = 1.5 * torch.sin(step * 0.1)

    obs, reward, terminated, truncated, info = env.step(actions)

    if step % 50 == 0:
        print(f"\n--- Step {step} ---")
        print("Environment 0 - Left leg joint positions:")
        for name, idx in left_leg_indices.items():
            pos = robot.data.joint_pos[0, idx].item()
            vel = robot.data.joint_vel[0, idx].item()
            effort = robot.data.applied_torque[0, idx].item()
            print(f"  {name:30s}: pos={pos:6.3f}  vel={vel:6.3f}  torque={effort:7.2f}")

        # Check contact
        print("\nContact sensor (env 0):")
        for i, body_name in enumerate(contact_sensor.data.body_names):
            if "ankle" in body_name.lower():
                force = contact_sensor.data.net_forces_w[0, i].norm().item()
                air_time = contact_sensor.data.current_air_time[0, i].item()
                contact_time = contact_sensor.data.current_contact_time[0, i].item()
                status = "GROUND" if contact_time > 0 else "AIR"
                print(f"  {body_name:30s}: force={force:6.1f}N  air={air_time:.2f}s  [{status}]")

        # Check if left leg actually moved
        if step > 0:
            knee_vel = robot.data.joint_vel[0, left_leg_indices["left_knee_joint"]].item()
            if abs(knee_vel) < 0.01:
                print("\n⚠️  WARNING: Left knee velocity near zero! Joint may be stuck!")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)
print("\n1. Did left leg joints move? (check pos/vel above)")
print("2. Did left ankle leave ground? (check air_time)")
print("3. If NO movement → actuator/URDF problem")
print("4. If movement but no air_time → contact sensor problem")

env.close()
