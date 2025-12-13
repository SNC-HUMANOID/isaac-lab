import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Path to URDF file
    urdf_file = '/home/code/Humanoid/src/humanoid_snc/urdf/Humanoid_10102025.urdf'
    
    # Read the URDF file
    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()
    
    return LaunchDescription([
        # Joint state publisher publishes zero positions for all joints
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        )
    ])