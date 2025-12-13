#!/usr/bin/env python3
import os
import subprocess
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    urdf_file = '/home/code/SNC Lab/isaac-lab/humanoid/src/g1_description/g1_23dof.urdf'
    package_path = '/home/code/SNC Lab/isaac-lab/humanoid/src/g1_description/'
    
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    # Replace relative mesh paths with absolute paths
    robot_description = robot_description.replace('filename="meshes/', f'filename="file://{package_path}meshes/')
    
    # Create wrapper script for joint_state_publisher_gui
    jsp_wrapper = '/home/code/SNC Lab/isaac-lab/humanoid/jsp_wrapper.sh'
    with open(jsp_wrapper, 'w') as f:
        f.write('''#!/bin/bash
unset GTK_PATH
unset LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0
exec /opt/ros/humble/lib/joint_state_publisher_gui/joint_state_publisher_gui "$@"
''')
    os.chmod(jsp_wrapper, 0o755)
    
    return LaunchDescription([
        ExecuteProcess(
            cmd=[jsp_wrapper],
            output='screen'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        ),
        ExecuteProcess(
            cmd=['/home/code/SNC Lab/isaac-lab/humanoid/rviz2_wrapper.sh'],
            output='screen'
        )
    ])
