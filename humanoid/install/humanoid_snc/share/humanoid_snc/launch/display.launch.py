import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_snc')
    urdf_file = os.path.join(pkg_share, 'urdf', 'Humanoid_abs.urdf')

    with open(urdf_file, 'r', encoding='utf-8') as urdf_handle:
        robot_description = urdf_handle.read()

    rviz_wrapper = '/home/code/SNC Lab/isaac-lab/humanoid/rviz2_wrapper.sh'

    return LaunchDescription([
        # Joint state publisher keeps TF tree populated
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),

        # Robot state publisher provides TF transforms from URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
        ),

        # RViz2 using wrapper to avoid snap conflicts
        ExecuteProcess(
            cmd=[rviz_wrapper],
            output='screen'
        )
    ])