#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import sys

class RobotArmController(Node):
    def __init__(self):
        super().__init__('robot_arm_controller')

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.left_joint_names = [
            'Body_leftarm1',
            'leftarm1_leftarm2',
            'leftarm2_leftarm3',
            'leftarm3_leftarm4',
            'leftarm4_leftarm5'
        ]

        self.right_joint_names = [
            'Body_rightarm1',
            'rightarm1_rightarm2',
            'rightarm2_rightarm3',
            'rightarm3_rightarm4',
            'rightarm4_rightarm5'
        ]

        self.left_joint_positions = [0.0] * 5
        self.right_joint_positions = [0.0] * 5

    def publish_joint_states(self):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()

        joint_state.name = self.left_joint_names + self.right_joint_names
        joint_state.position = self.left_joint_positions + self.right_joint_positions

        self.joint_pub.publish(joint_state)

    def set_left_joint_angles(self, angles):
        if len(angles) != 5:
            self.get_logger().error("Need exactly 5 joint angles for left arm")
            return
        self.left_joint_positions = angles
        self.publish_joint_states()

    def set_right_joint_angles(self, angles):
        if len(angles) != 5:
            self.get_logger().error("Need exactly 5 joint angles for right arm")
            return
        self.right_joint_positions = angles
        self.publish_joint_states()

    def control_individual_joint(self, arm, joint_index, angle):
        if arm.lower() == 'left':
            if 0 <= joint_index < 5:
                self.left_joint_positions[joint_index] = angle
            else:
                self.get_logger().error("Invalid joint index for left arm (0-4)")
        elif arm.lower() == 'right':
            if 0 <= joint_index < 5:
                self.right_joint_positions[joint_index] = angle
            else:
                self.get_logger().error("Invalid joint index for right arm (0-4)")
        else:
            self.get_logger().error("Invalid arm specified (use 'left' or 'right')")
        self.publish_joint_states()


def main(args=None):
    rclpy.init(args=args)
    controller = RobotArmController()
    rate = controller.create_rate(10)

    try:
        while rclpy.ok():
            print("\nRobot Arm Control Options:")
            print("1. Set all left arm joints")
            print("2. Set all right arm joints")
            print("3. Control individual joint")
            print("4. Reset to home position")
            print("5. Exit")

            choice = input("Enter your choice (1-5): ")

            if choice == '1':
                angles = [float(x) for x in input("Enter 5 angles (radians) separated by spaces: ").split()]
                controller.set_left_joint_angles(angles)
            elif choice == '2':
                angles = [float(x) for x in input("Enter 5 angles (radians) separated by spaces: ").split()]
                controller.set_right_joint_angles(angles)
            elif choice == '3':
                arm = input("Which arm (left/right)? ").strip()
                joint = int(input("Which joint (0-4)? "))
                angle = float(input("Angle (radians): "))
                controller.control_individual_joint(arm, joint, angle)
            elif choice == '4':
                controller.set_left_joint_angles([0.0]*5)
                controller.set_right_joint_angles([0.0]*5)
                print("Reset to home position")
            elif choice == '5':
                break
            else:
                print("Invalid choice")

            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
