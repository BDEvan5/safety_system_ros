from safety_system_ros.Supervisor import Supervisor
from safety_system_ros.BaseNode import BaseNode
from safety_system_ros.Planners.TrainingAgent import TestVehicle

import rclpy
import time
import numpy as np

from safety_system_ros.utils.util_functions import *


class TestingNode(BaseNode):
    def __init__(self):
        super().__init__('car_tester')

        self.declare_parameter('n_laps')
        self.declare_parameter('supervision')
        self.declare_parameter('agent_name')
        self.declare_parameter('map_name')

        map_name = self.get_parameter('map_name').value
        agent_name = self.get_parameter('agent_name').value

        self.planner = TestVehicle(self.conf, agent_name)
        self.get_logger().info(self.planner.name)

        self.supervision = self.get_parameter('supervision').value
        if self.supervision:
            self.supervisor = Supervisor(self.conf, map_name)

        self.n_laps = self.get_parameter('n_laps').value
        self.get_logger().info(f"Number of test laps laps: {self.n_laps}")

    def calculate_action(self, observation):
        action = self.planner.plan(observation)
        if self.supervision: 
            return self.supervisor.supervise(observation['state'], action)
        return action

    def lap_complete_callback(self):
        self.get_logger().info(f"Lap complee: {self.current_lap_time}")
        if self.supervision:
            self.get_logger().info(f"Interventions: {self.supervisor.interventions}")



def main(args=None):
    rclpy.init(args=args)
    node = TestingNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


