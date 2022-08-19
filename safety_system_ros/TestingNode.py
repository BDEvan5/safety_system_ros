from safety_system_ros.DriveNode import DriveNode
from safety_system_ros.Planners.TrainingAgent import TestVehicle

import rclpy
import time
import numpy as np

from safety_system_ros.utils.util_functions import *
from safety_system_ros.utils.LapLogger import LapLogger

class VehicleStateHistory:
    def __init__(self, vehicle_name, map_name):
        self.vehicle_name = vehicle_name
        directory = "/home/benjy/sim_ws/src/safety_system_ros/"
        self.path = directory + "Data/Vehicles/" + vehicle_name+ "/"
        self.states = []
        self.actions = []
        self.map_name = map_name

    def add_state(self, state):
        self.states.append(state)
    
    def add_action(self, action):
        self.actions.append(action)
    
    def save_history(self, lap_n=0):
        states = np.array(self.states)
        # self.actions.append(np.array([0, 0])) # last action to equal lengths
        actions = np.array(self.actions)

        lap_history = np.concatenate((states, actions), axis=1)

        np.save(self.path + f"Lap_{lap_n}_{self.map_name}_history_{self.vehicle_name}.npy", lap_history)

        self.states = []
        self.actions = []


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
        self.vehicles_state_history = VehicleStateHistory(agent_name, "levine_blocked")

        self.steering_actions = []

    def calculate_action(self, observation):
        action = self.agent.plan(observation)
        self.steering_actions.append(action[0])
        self.vehicles_state_history.add_state(observation['state'])
        self.vehicles_state_history.add_action(action)
        return action

    def lap_complete_callback(self):
        self.get_logger().info(f"Lap complee: {self.current_lap_time}")
        self.vehicles_state_history.save_history(self.lap_count)
        # np.save(self.agent.path + "/steering_actions.npy", self.steering_actions)


def main(args=None):
    rclpy.init(args=args)
    node = AgentTester()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


