
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    ld = LaunchDescription()

    # config = os.path.join(
    #     get_package_share_directory('safety_system_ros'),
    #     'config',
    #     'testing_params.yaml'
    # )
    config = os.path.join(
        "/home/benjy/sim_ws/src/safety_system_ros/",
        'config',
        'testing_params.yaml'
    )

    # config_dict = yaml.safe_load(open(config, 'r'))
    # print(f"Config Dict: {config_dict}")

    testing_node = Node(
        package='safety_system_ros',
        executable='car_tester',
        name='car_tester',
        # parameters=[{'n_laps': 1}]
        parameters=[config]
    )

    ld.add_action(testing_node)
    return ld

# generate_launch_description()
