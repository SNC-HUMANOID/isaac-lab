from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gazebo_ros',
            executable='gazebo',
            name='gazebo',
            output='screen',
            arguments=['-s', 'libgazebo_ros_factory.so', LaunchConfiguration('world')],
            parameters=[{'robot_description': LaunchConfiguration('robot_description')}],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': LaunchConfiguration('robot_description')}],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
        ),
    ])