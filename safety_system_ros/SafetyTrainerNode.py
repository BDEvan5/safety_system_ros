import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped


import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from safety_system_ros.utils.util_functions import *
from safety_system_ros.utils.Dynamics import *
from copy import copy

from safety_system_ros.Supervisor import Supervisor, LearningSupervisor
from safety_system_ros.Planners.TrainingAgent import TrainVehicle, TestVehicle
from safety_system_ros.BaseNode import BaseNode

class SafetyTrainer(BaseNode):
    def __init__(self):
        super().__init__("safety_trainer")
        self.declare_parameter('agent_name')
        self.declare_parameter('map_name')
        self.declare_parameter('n_laps')


        agent_name = self.get_parameter('agent_name').value
        map_name = self.get_parameter('map_name').value
        self.n_laps = self.get_parameter('n_laps').value

        self.planner = TrainVehicle(self.conf, agent_name) 
        self.supervisor = LearningSupervisor(self.planner, self.conf, map_name)

        # this is the asyn training frequency: consider making parameter
        self.training_timer = self.create_timer(0.1, self.training_callback)

        self.get_logger().info("SafetyTrainer initialized")

    def training_callback(self):
        self.planner.agent.train(2)

    def calculate_action(self, observation):
        safe_action = self.supervisor.plan(observation) 

        return safe_action

    def save_data_callback(self):
        self.planner.agent.save(self.planner.path)
        self.planner.t_his.print_update(True)
        self.planner.t_his.save_csv_data()
        self.supervisor.save_intervention_list()

    def lap_complete_callback(self):
        self.get_logger().info(f"Interventions: {self.supervisor.ep_interventions}")
        self.supervisor.lap_complete(self.current_lap_time)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyTrainer()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

