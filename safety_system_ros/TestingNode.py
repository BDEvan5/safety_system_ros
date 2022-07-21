from safety_system_ros.DriveNode import DriveNode
from safety_system_ros.Planners.TrainingAgent import TestVehicle

import rclpy
import time
import numpy as np

from safety_system_ros.utils.util_functions import *
from safety_system_ros.utils.LapLogger import LapLogger


class AgentTester(DriveNode):
    def __init__(self):
        super().__init__('agent_tester')

        self.declare_parameter('n_laps')
        self.declare_parameter('agent_name')
        agent_name = self.get_parameter('agent_name').value

        self.planner = TestVehicle(self.conf, agent_name)
        self.get_logger().info(self.planner.name)

        self.n_laps = self.get_parameter('n_laps').value
        self.get_logger().info(f"Number of test laps laps: {self.n_laps}")

        self.agent = TestVehicle(self.conf, agent_name)

        self.steering_actions = []

    def calculate_action(self, observation):
        action = self.agent.plan(observation)
        self.steering_actions.append(action[0])
        return action

    def lap_complete_callback(self):
        self.get_logger().info(f"Lap complee: {self.current_lap_time}")
        np.save(self.agent.path + "/steering_actions.npy", self.steering_actions)


def main(args=None):
    rclpy.init(args=args)
    node = AgentTester()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


