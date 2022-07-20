
from launch import LaunchDescription
import launch
from launch_ros.actions import Node

import os, yaml 

def generate_launch_description():
    config = os.path.join(
    "/home/benjy/sim_ws/src/safety_system_ros/",
    'config',
    'bag_files.yaml'
)
    localize_config_dict = yaml.safe_load(open(config, 'r'))
    bag_name = localize_config_dict['bag_extractor']['ros__parameters']['bag_name']
    rate = localize_config_dict['bag_extractor']['ros__parameters']['rate']

    ld = LaunchDescription()

    # bag_path = '/home/benjy/Documents/USA Safety Data/' + bag_name + ''
    # action = launch.actions.ExecuteProcess( cmd=['ros2', 'bag', 'play', bag_path, '-r', rate],  output='screen'    )
    
    # node = Node()
    testing_node = Node(
        package='safety_system_ros',
        executable='bag_extractor',
        name='bag_extractor',
        parameters=[{'bag_name': bag_name}],
    )

    ld.add_action( testing_node )
    # ld.add_action( action )

    return ld

