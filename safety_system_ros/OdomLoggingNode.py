from abc import abstractmethod
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped


import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from safety_system_ros.utils.Dynamics import *
from safety_system_ros.utils.util_functions import *
from copy import copy





class LapLoggingNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        
        # current vehicle state
        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0

        self.lap_count = 0 
        self.n_laps = None

        self.logger = None

        odom_topic = "pf/pose/odom"
        # odom_topic = "ego_racecar/odom"
        self.odom_subscriber = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        abs_path = "/home/nvidia/f1tenth_ws/src/safety_system_ros/"
        # abs_path = "/home/benjy/sim_ws/src/safety_system_ros/"
        # self.path = abs_path + "Data/Logs/"
        self.path = abs_path + "Data/"  # activate this after teting
        self.current_log_file = None

        self.lap = 0

        self.info_log_file = self.path + f"InfoLogging.txt"
    
        with open(self.info_log_file, "w") as f:
            print(f"{self.info_log_file} created")

        self.open_log_file()

    def open_log_file(self):
        self.current_log_file = f"{self.lap}_log_file.csv"
        with open(self.path + self.current_log_file, "w") as f:
            print(f"{self.current_log_file} created")
            # f.write("act_steer, act_vel, pos_x, pos_y, theta, v\n")

    def reset_logging(self):
        self.lap += 1 
        self.open_log_file()

    def write_env_log(self, data):
        with open(self.path + self.current_log_file, "a") as f:
            f.write(data + "\n")

    def write_info_log(self, log_text):
        with open(self.info_log_file, "a") as f:
            f.write(log_text + "\n")


    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.position = np.array([position.x, position.y])
        self.velocity = msg.twist.twist.linear.x

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        theta = z * np.pi / 180
        self.theta = copy(theta)

        self.write_env_log(f"{self.position[0]}, {self.position[1]}, {self.theta}, {self.velocity}")
