
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()

    pp_node = Node(
        package='safety_system_ros',
        executable='pure_pursuit',
        name='pp_planner'
    )

    ld.add_action(pp_node)
    return ld
