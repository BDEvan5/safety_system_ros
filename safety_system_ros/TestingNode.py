from safety_system_ros.Supervisor import Supervisor
from safety_system_ros.BaseNode import BaseNode
from safety_system_ros.Planners.TrainingAgent import TestVehicle
from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner

import rclpy
import time
import numpy as np

from safety_system_ros.utils.util_functions import *


class TestingNode(BaseNode):
    def __init__(self):
        super().__init__('car_tester')

        self.declare_parameter('n_laps')
        self.declare_parameter('supervision')
        self.declare_parameter('planner')
        self.declare_parameter('agent_name')
        self.declare_parameter('map_name')

        map_name = self.get_parameter('map_name').value

        self.planner = PurePursuitPlanner(self.conf, map_name)
        self.get_logger().info(self.planner.name)
        self.get_logger().info(str(self.planner.trajectory.waypoints))

        self.supervision = self.get_parameter('supervision').value
        # self.supervisor = Supervisor(self.conf, map_name)

        self.n_laps = self.get_parameter('n_laps').value
        self.get_logger().info(f"Param laps: {self.n_laps}")

    def calculate_action(self, observation):
        action = self.planner.plan(observation)
        if self.supervision: 
            return self.supervisor.supervise(observation['state'], action)
        return action

    def lap_complete_callback(self):
        print(f"Lap complee: {self.current_lap_time}")
        # print(f"Interventions: {self.supervisor.interventions}")



def main(args=None):
    rclpy.init(args=args)
    node = TestingNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


