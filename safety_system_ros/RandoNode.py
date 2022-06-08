from safety_system_ros.Supervisor import Supervisor
from safety_system_ros.BaseNode import BaseNode

import rclpy
import time
import numpy as np

from safety_system_ros.utils.util_functions import *

class RandomPlanner:
    def __init__(self, conf, name="RandoPlanner"):
        self.d_max = conf.max_steer # radians  
        self.name = name
        self.speed = conf.vehicle_speed

    def plan(self, pos):
        steering = np.random.uniform(-self.d_max, self.d_max)
        return np.array([steering, self.speed])



class TestingNode(BaseNode):
    def __init__(self):
        super().__init__('car_tester')

        self.declare_parameter('n_laps')
        self.declare_parameter('map_name')

        map_name = self.get_parameter('map_name').value

        self.planner = RandomPlanner(self.conf)
        self.get_logger().info(self.planner.name)

        self.get_logger().info("Supervision always enabled")
        self.supervisor = Supervisor(self.conf, map_name)

        self.n_laps = self.get_parameter('n_laps').value
        self.get_logger().info(f"Number of laps to run: {self.n_laps}")

    def calculate_action(self, observation):
        action = self.planner.plan(observation)
        return self.supervisor.supervise(observation['state'], action)

    def lap_complete_callback(self):
        print(f"Lap complee: {self.current_lap_time}")
        print(f"Interventions: {self.supervisor.interventions}")



def main(args=None):
    rclpy.init(args=args)
    node = TestingNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


