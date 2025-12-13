#!/usr/bin/env python3
"""
ROS 2 node for running IsaacLab-trained humanoid policy on real robot.

This node:
1. Subscribes to joint states and IMU data from the robot
2. Builds observations matching the training environment
3. Runs the policy inference
4. Publishes joint position commands

Usage:
    # First export the policy
    python ros2/export_policy_torchscript.py \
        --checkpoint logs/rsl_rl/humanoid_10102025/2025-11-13_08-57-58/model_9050.pt \
        --output ros2/humanoid_policy.pt
    
    # Then run this node
    ros2 run your_package humanoid_policy_node \
        --ros-args -p policy_path:=ros2/humanoid_policy.pt
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import torch
import numpy as np
from collections import deque

# ROS 2 messages
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


class HumanoidPolicyNode(Node):
    """ROS 2 node for humanoid policy inference."""
    
    def __init__(self):
        super().__init__('humanoid_policy_node')
        
        # Declare parameters
        self.declare_parameter('policy_path', 'ros2/humanoid_policy.pt')
        self.declare_parameter('control_rate', 200.0)  # Hz
        self.declare_parameter('enable_safety', True)
        self.declare_parameter('max_joint_vel', 10.0)  # rad/s
        self.declare_parameter('action_scale', 0.25)
        
        # Get parameters
        policy_path = self.get_parameter('policy_path').value
        control_rate = self.get_parameter('control_rate').value
        self.enable_safety = self.get_parameter('enable_safety').value
        self.max_joint_vel = self.get_parameter('max_joint_vel').value
        self.action_scale = self.get_parameter('action_scale').value
        
        self.get_logger().info(f"Loading policy from: {policy_path}")
        
        # Load policy
        try:
            self.policy = torch.jit.load(policy_path, map_location='cpu')
            self.policy.eval()
            self.get_logger().info("✓ Policy loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            raise
        
        # Robot configuration - ADJUST THESE FOR YOUR ROBOT
        self.joint_names = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint',
            'left_hip_roll_joint', 'right_hip_roll_joint',
            'left_hip_yaw_joint', 'right_hip_yaw_joint',
            'left_knee_joint', 'right_knee_joint',
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
            'left_ankle_roll_joint', 'right_ankle_roll_joint',
            'waist_joint',
            'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
            'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
            'left_elbow_joint', 'right_elbow_joint',
        ]
        
        # Default joint positions (standing pose)
        self.default_joint_pos = np.array([
            0.0, 0.0,  # hip_pitch
            0.0, 0.0,  # hip_roll
            0.0, 0.0,  # hip_yaw
            0.0, 0.0,  # knee
            0.0, 0.0,  # ankle_pitch
            0.0, 0.0,  # ankle_roll
            0.0,       # waist
            0.0, 0.0,  # shoulder_pitch
            0.0, 0.0,  # shoulder_roll
            0.0, 0.0,  # shoulder_yaw
            0.0, 0.0,  # elbow
        ])
        
        # Joint limits [lower, upper] in radians
        self.joint_limits = np.array([
            [-1.57, 1.57], [- 1.57, 1.57],  # hip_pitch
            [-0.5, 0.5], [-0.5, 0.5],        # hip_roll
            [-0.5, 0.5], [-0.5, 0.5],        # hip_yaw
            [-2.0, 0.1], [-2.0, 0.1],        # knee
            [-0.5, 0.5], [-0.5, 0.5],        # ankle_pitch
            [-0.3, 0.3], [-0.3, 0.3],        # ankle_roll
            [-0.5, 0.5],                     # waist
            [-1.57, 1.57], [-1.57, 1.57],   # shoulder_pitch
            [-0.5, 0.5], [-0.5, 0.5],        # shoulder_roll
            [-0.5, 0.5], [-0.5, 0.5],        # shoulder_yaw
            [-1.57, 0.0], [-1.57, 0.0],     # elbow
        ])
        
        # State variables
        self.last_joint_state = None
        self.last_imu = None
        self.command = np.array([0.0, 0.0, 0.0])  # [vx, vy, yaw_rate]
        self.last_action = np.zeros(len(self.joint_names))
        self.action_history = deque(maxlen=3)  # For filtering
        
        # QoS profile for real-time data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.sub_joint_state = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile
        )
        
        self.sub_imu = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            qos_profile
        )
        
        self.sub_cmd_vel = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Publishers
        self.pub_joint_cmd = self.create_publisher(
            Float32MultiArray,
            '/joint_position_command',
            10
        )
        
        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / control_rate,
            self.control_loop
        )
        
        self.get_logger().info(f"Node initialized at {control_rate} Hz")
        self.get_logger().info(f"Waiting for sensor data...")
    
    def joint_state_callback(self, msg):
        """Callback for joint state messages."""
        self.last_joint_state = msg
    
    def imu_callback(self, msg):
        """Callback for IMU messages."""
        self.last_imu = msg
    
    def cmd_vel_callback(self, msg):
        """Callback for velocity command messages."""
        self.command[0] = msg.linear.x
        self.command[1] = msg.linear.y
        self.command[2] = msg.angular.z
    
    def build_observation(self):
        """
        Build observation vector from robot state.
        
        IMPORTANT: This must match EXACTLY the observation structure used during training!
        
        Observation structure for velocity environment (75 dimensions):
        1. Base linear velocity (3) - estimated from joint velocities or IMU integration
        2. Base angular velocity (3) - from IMU
        3. Projected gravity vector (3) - from IMU
        4. Velocity commands (3) - target velocities
        5. Joint positions relative (21) - normalized to default
        6. Joint velocities (21) - from joint states
        7. Previous actions (21) - last policy output
        Total: 75 elements
        """
        if self.last_joint_state is None or self.last_imu is None:
            return None
        
        try:
            # Extract joint states in correct order
            joint_pos = np.zeros(len(self.joint_names))
            joint_vel = np.zeros(len(self.joint_names))
            
            for i, name in enumerate(self.joint_names):
                try:
                    idx = self.last_joint_state.name.index(name)
                    joint_pos[i] = self.last_joint_state.position[idx]
                    joint_vel[i] = self.last_joint_state.velocity[idx]
                except (ValueError, IndexError):
                    self.get_logger().warn(f"Joint {name} not found in joint_states")
                    return None
            
            # 1. Base linear velocity (3) - estimate from IMU or set to zero for now
            # In real robot, you may need odometry or state estimation
            base_lin_vel = np.zeros(3)  # TODO: Integrate from IMU or use odometry
            
            # 2. Base angular velocity from IMU (3)
            base_ang_vel = np.array([
                self.last_imu.angular_velocity.x,
                self.last_imu.angular_velocity.y,
                self.last_imu.angular_velocity.z
            ])
            
            # 3. Projected gravity vector from IMU (3)
            gravity_vec = np.array([
                self.last_imu.linear_acceleration.x,
                self.last_imu.linear_acceleration.y,
                self.last_imu.linear_acceleration.z
            ])
            gravity_vec = gravity_vec / (np.linalg.norm(gravity_vec) + 1e-8)
            
            # 4. Velocity commands (3)
            velocity_commands = self.command
            
            # 5. Joint positions relative to default (21)
            joint_pos_rel = joint_pos - self.default_joint_pos
            
            # 6. Joint velocities (21)
            joint_vel_rel = joint_vel
            
            # 7. Previous actions (21)
            prev_actions = self.last_action
            
            # Concatenate observation - ORDER MATTERS!
            obs = np.concatenate([
                base_lin_vel,       # 3
                base_ang_vel,       # 3
                gravity_vec,        # 3
                velocity_commands,  # 3
                joint_pos_rel,      # 21
                joint_vel_rel,      # 21
                prev_actions,       # 21
            ])
            
            assert len(obs) == 75, f"Observation must be 75 dims, got {len(obs)}"
            
            return obs
            
        except Exception as e:
            self.get_logger().error(f"Error building observation: {e}")
            return None
    
    def clip_action(self, action):
        """Apply safety limits to action."""
        # Clip action magnitude
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action filter (moving average)
        self.action_history.append(action)
        if len(self.action_history) > 1:
            action = np.mean(self.action_history, axis=0)
        
        return action
    
    def action_to_joint_targets(self, action):
        """
        Convert policy action to joint position targets.
        
        Args:
            action: Policy output (normalized)
        
        Returns:
            target_positions: Joint position commands [rad]
        """
        # Scale action
        action_scaled = action * self.action_scale
        
        # Compute target positions
        target_pos = self.default_joint_pos + action_scaled
        
        # Clip to joint limits
        target_pos = np.clip(
            target_pos,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        )
        
        # Safety check: limit velocity
        if self.enable_safety and self.last_joint_state is not None:
            current_pos = np.zeros(len(self.joint_names))
            for i, name in enumerate(self.joint_names):
                try:
                    idx = self.last_joint_state.name.index(name)
                    current_pos[i] = self.last_joint_state.position[idx]
                except (ValueError, IndexError):
                    pass
            
            # Limit change per timestep
            dt = 1.0 / self.get_parameter('control_rate').value
            max_change = self.max_joint_vel * dt
            delta = target_pos - current_pos
            delta = np.clip(delta, -max_change, max_change)
            target_pos = current_pos + delta
        
        return target_pos
    
    def control_loop(self):
        """Main control loop - runs at control_rate Hz."""
        # Build observation
        obs = self.build_observation()
        if obs is None:
            return
        
        # Convert to torch tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        # Run policy inference
        try:
            with torch.no_grad():
                action_tensor = self.policy(obs_tensor)
            action = action_tensor.cpu().numpy()
        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            return
        
        # Apply safety filters
        if self.enable_safety:
            action = self.clip_action(action)
        
        # Convert to joint targets
        target_positions = self.action_to_joint_targets(action)
        
        # Publish commands
        msg = Float32MultiArray()
        msg.data = target_positions.tolist()
        self.pub_joint_cmd.publish(msg)
        
        self.last_action = action
    
    def shutdown(self):
        """Cleanup on shutdown."""
        self.get_logger().info("Shutting down policy node...")
        # Send zero command or default position
        msg = Float32MultiArray()
        msg.data = self.default_joint_pos.tolist()
        self.pub_joint_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = HumanoidPolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
