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
        conf = load_conf("config_file")
        super().__init__("safety_trainer", conf)

        self.planner = TrainVehicle("SafetyTrainingAgent_2", conf) 
        self.supervisor = LearningSupervisor(self.planner, conf)

        self.n_laps = 8

        self.training_timer = self.create_timer(0.1, self.training_callback)

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
        print(f"Interventions: {self.supervisor.ep_interventions}")
        self.supervisor.lap_complete(self.current_lap_time)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyTrainer()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

