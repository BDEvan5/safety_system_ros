
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        "/home/nvidia/f1tenth_ws/src/safety_system_ros/",
        'config',
        'testing_params.yaml'
    )

    testing_node = Node(
        package='safety_system_ros',
        executable='gap_follow',
        name='gap_follow',
        parameters=[config]
    )

    ld.add_action(testing_node)
    return ld

