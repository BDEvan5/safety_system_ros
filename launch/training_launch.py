
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()

    testing_node = Node(
        package='safety_system_ros',
        executable='safety_trainer',
        name='safety_trainer'
    )

    ld.add_action(testing_node)
    return ld
