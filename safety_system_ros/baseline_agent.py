

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan

from safety_system_ros.TD3 import TD3

from matplotlib import pyplot as plt
import math
import numpy as np
from numba import njit
import csv
import os


class BaselineDRL(Node):
    def __init__(self):
        super().__init__('baseline_agent')

        self.v_min_plan = 1
        self.wheelbase =  0.33
        self.max_steer = 0.4
        self.vehicle_speed = 2.0

        self.n_beams = 27

        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0

        self.agent = TD3(self.n_beams, 1, 1, "DRL_agent")

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_timer = self.create_timer(0.1, self.send_cmd_msg)

        self.scan_subscriber = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

    def scan_callback(self, msg):
        scan = np.array(msg.ranges)
        inds = np.arrange(0, 1080, 40)
        self.scan = scan[inds]


    def send_cmd_msg(self):

        

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.vehicle_speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_publisher.publish(drive_msg)




