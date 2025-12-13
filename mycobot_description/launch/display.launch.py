from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get URDF path
    
    # Read URDF contents
    with open('/home/code/AMR_WS/src/mycobot_description/urdf/humanoid/humanoid.urdf', 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # Robot State Publisher (loads URDF)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'publish_frequency': 30.0  # Hz
            }]
        ),
        
        # # Joint State Publisher (GUI for testing)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),
        
        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/home/code/Desktop/rviz1.rviz']
        )
    ])