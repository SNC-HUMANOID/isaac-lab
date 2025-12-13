#!/usr/bin/env python3
"""
Test policy without ROS 2 - simulates sensor data for debugging.

Usage:
    python ros2/test_policy_simulation.py --policy ros2/humanoid_policy.pt
"""

import argparse
import time
import numpy as np
import torch


def simulate_robot_state(step):
    """Simulate robot sensor data."""
    # Simulate 21 joints
    joint_pos = np.sin(step * 0.01) * 0.1 * np.ones(21)
    joint_vel = np.cos(step * 0.01) * 0.5 * np.ones(21)
    
    # Simulate IMU
    base_ang_vel = np.array([0.0, 0.0, 0.0])
    gravity_vec = np.array([0.0, 0.0, -1.0])  # Normalized gravity
    
    # Simulate command
    command = np.array([0.5, 0.0, 0.0])  # Walk forward 0.5 m/s
    
    return joint_pos, joint_vel, base_ang_vel, gravity_vec, command


def build_observation(joint_pos, joint_vel, base_ang_vel, gravity_vec, command, default_pos, prev_actions):
    """
    Build observation vector matching training format (75 dimensions).
    
    Order must match velocity_env_cfg.py:
    1. Base linear velocity (3)
    2. Base angular velocity (3)
    3. Projected gravity (3)
    4. Velocity commands (3)
    5. Joint positions relative (21)
    6. Joint velocities (21)
    7. Previous actions (21)
    """
    base_lin_vel = np.zeros(3)  # Placeholder - would come from odometry in real robot
    joint_pos_rel = joint_pos - default_pos
    
    obs = np.concatenate([
        base_lin_vel,   # 3
        base_ang_vel,   # 3
        gravity_vec,    # 3
        command,        # 3
        joint_pos_rel,  # 21
        joint_vel,      # 21
        prev_actions,   # 21
    ])
    
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to TorchScript policy")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--rate", type=float, default=200.0, help="Control rate (Hz)")
    args = parser.parse_args()
    
    print(f"Loading policy from: {args.policy}")
    policy = torch.jit.load(args.policy, map_location='cpu')
    policy.eval()
    print("✓ Policy loaded\n")
    
    # Robot configuration
    default_joint_pos = np.zeros(21)
    action_scale = 0.25
    
    print(f"Running simulation at {args.rate} Hz for {args.steps} steps...")
    print("Press Ctrl+C to stop\n")
    
    dt = 1.0 / args.rate
    action_history = []
    prev_actions = np.zeros(21)  # Initialize previous actions
    
    try:
        for step in range(args.steps):
            start_time = time.time()
            
            # Simulate sensor data
            joint_pos, joint_vel, base_ang_vel, gravity_vec, command = simulate_robot_state(step)
            
            # Build observation
            obs = build_observation(joint_pos, joint_vel, base_ang_vel, gravity_vec, command, default_joint_pos, prev_actions)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            # Run policy
            with torch.no_grad():
                action_tensor = policy(obs_tensor)
            action = action_tensor.cpu().numpy()
            
            # Compute target positions
            target_pos = default_joint_pos + action_scale * action
            
            # Store action
            action_history.append(action)
            prev_actions = action  # Update previous actions for next step
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}/{args.steps}")
                print(f"  Obs: {obs[:5]} ... (first 5)")
                print(f"  Action: {action[:5]} ... (first 5)")
                print(f"  Target pos: {target_pos[:5]} ... (first 5)")
                print()
            
            # Sleep to maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
            
            if elapsed > dt:
                print(f"Warning: Control loop too slow! ({elapsed*1000:.1f} ms > {dt*1000:.1f} ms)")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    # Statistics
    if action_history:
        actions = np.array(action_history)
        print(f"\n{'='*60}")
        print("Action Statistics:")
        print(f"{'='*60}")
        print(f"  Mean: {actions.mean(axis=0)[:5]} ... (first 5)")
        print(f"  Std:  {actions.std(axis=0)[:5]} ... (first 5)")
        print(f"  Min:  {actions.min():.3f}")
        print(f"  Max:  {actions.max():.3f}")
        print(f"\n✓ Simulation complete!")


if __name__ == "__main__":
    main()
