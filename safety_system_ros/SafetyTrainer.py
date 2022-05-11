import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from safety_system_ros.utils import *
# from safety_system_ros.PurePursuitPlanner import PurePursuitPlanner 

from safety_system_ros.Dynamics import *
from safety_system_ros.PurePursuitPlanner import PurePursuitPlanner
from copy import copy

from safety_system_ros.Supervisor import Supervisor, LearningSupervisor
from safety_system_ros.TrainingAgent import TrainVehicle, TestVehicle

class Trainer(Node):
    def __init__(self):
        super().__init__('safety_trainer')
        
        conf = load_conf("config_file")

        self.planner = TrainVehicle("SafetyTrainingAgent", conf) 
        self.supervisor = LearningSupervisor(self.planner, conf)

        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0
        self.steering_angle = 0.0
        self.scan = np.zeros(27)

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_timer = self.create_timer(0.03, self.send_cmd_msg)

        self.odom_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.current_drive_sub = self.create_subscription(AckermannDrive, 'ego_racecar/current_drive', self.current_drive_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

    def current_drive_callback(self, msg):
        self.steering_angle = msg.steering_angle

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.position = np.array([position.x, position.y])
        self.velocity = msg.twist.twist.linear.x

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        theta = z * np.pi / 180
        self.theta = copy(theta)

    def scan_callback(self, msg):
        scan = np.array(msg.ranges)
        inds = np.arange(0, 1080, 40)
        scan = scan[inds]

        self.scan = scan

    def send_cmd_msg(self):
        observation = {}
        observation["scan"] = self.scan
        observation['linear_vel_x'] = self.velocity
        observation['steering_delta'] = self.steering_angle
        state = np.array([self.position[0], self.position[1], self.theta, self.velocity, self.steering_angle])
        observation['state'] = state
        observation['reward'] = 0.0

        safe_action = self.supervisor.plan(observation) 

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(safe_action[1])
        drive_msg.drive.steering_angle = float(safe_action[0])
        self.drive_publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Trainer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

