from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # ระบุ path ไปยังไฟล์ URDF
    urdf_path = '/home/code/ros2_ws/src/mycobot_description/urdf/humanoid/humanoid.urdf'
    
    # อ่านเนื้อหา URDF
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # Robot State Publisher (โหลด URDF)
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
        
        # Joint State Publisher GUI (สำหรับทดสอบ)
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
            arguments=['-d', '/home/code/Desktop/rviz1.rviz'],
            output='screen'
        ),
        # แทนที่ส่วน Gazebo ด้วย:
        ExecuteProcess(
            cmd=['ign gazebo -v 4'],
            output='screen'
        ),

        # ในส่วน spawn entity ให้แก้ไขเป็น:
        Node(
            package='ros_ign_gazebo',
            executable='create',
            arguments=[
                '-topic', '/robot_description',
                '-name', 'humanoid_robot',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.1'
            ],
            output='screen',
            # เพิ่ม delay 10 วินาทีและรอจนกว่า Gazebo จะพร้อม
            prefix='bash -c "sleep 10; while ! ros2 topic list | grep -q \'/world/default\'; do sleep 1; done; $0 $@"'
        )
      
    ])