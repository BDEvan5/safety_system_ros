import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped


import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from safety_system_ros.utils.utils import *
from safety_system_ros.utils.Dynamics import *
from copy import copy

from safety_system_ros.Supervisor import Supervisor
from safety_system_ros.TrainingAgent import TestVehicle

class Tester(Node):
    def __init__(self):
        super().__init__('safety_tester')
        
        conf = load_conf("config_file")

        self.planner = TestVehicle("SafetyTrainingAgent", conf) 

        self.supervisor = Supervisor(conf)

        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0
        self.steering_angle = 0.0
        self.scan = np.zeros(27)

        self.lap_counts = 0
        self.lap_times = 0.0 
        self.toggle_list = 0
        self.near_start = True
        self.current_lap_time = 0.0
        self.running = False

        self.lap_count = 0 
        self.n_laps = 2

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_timer = self.create_timer(0.03, self.send_cmd_msg)

        self.odom_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.current_drive_sub = self.create_subscription(AckermannDrive, 'ego_racecar/current_drive', self.current_drive_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        self.ego_reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose', 10)

    def current_drive_callback(self, msg):
        self.steering_angle = msg.steering_angle

    def training_callback(self):
        self.planner.agent.train(2)

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
        if not self.running:
            return

        done = self.check_lap_done(self.position)
        if done:
            print(f"Run Complete in time: {self.current_lap_time}")
            print(f"Number of interventions: {self.supervisor.interventions}")
            # self.supervisor.lap_complete(self.current_lap_time)

            self.data_reset()
            self.lap_count += 1

            if self.lap_count == self.n_laps:
                self.running = False
                self.ego_reset()
                self.destroy_node()

        observation = {}
        observation["scan"] = self.scan
        observation['linear_vel_x'] = self.velocity
        observation['steering_delta'] = self.steering_angle
        state = np.array([self.position[0], self.position[1], self.theta, self.velocity, self.steering_angle])
        observation['state'] = state
        observation['reward'] = 0.0

        action = self.planner.plan(observation)
        safe_action = self.supervisor.supervise(state, action)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(safe_action[1])
        drive_msg.drive.steering_angle = float(safe_action[0])
        self.drive_publisher.publish(drive_msg)

        self.current_lap_time += 0.03 #this might be inaccurate due to processing

    def check_lap_done(self, position):
        start_x = 0
        start_y = 0 
        start_theta = 0
        start_rot = np.array([[np.cos(-start_theta), -np.sin(-start_theta)], [np.sin(-start_theta), np.cos(-start_theta)]])

        poses_x = np.array(position[0])-start_x
        poses_y = np.array(position[1])-start_y
        delta_pt = np.dot(start_rot, np.stack((poses_x, poses_y), axis=0))

        dist2 = delta_pt[0]**2 + delta_pt[1]**2
        closes = dist2 <= 0.2
        if closes and not self.near_start:
            self.near_start = True
            self.toggle_list += 1
        elif not closes and self.near_start:
            self.near_start = False
            self.toggle_list += 1
            # print(self.toggle_list)
        self.lap_counts = self.toggle_list // 2
        if self.toggle_list < 4:
            self.lap_times = self.current_lap_time
        
        done = self.toggle_list >= 2

        return done

    def data_reset(self):
        self.current_lap_time = 0.0
        self.num_toggles = 0
        self.near_start = True
        self.toggle_list = 0

    def ego_reset(self):
        msg = PoseWithCovarianceStamped() 

        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        self.ego_reset_pub.publish(msg)

        print("Finished Resetting")

    def run_lap(self):
        self.ego_reset()

        self.current_lap_time = 0.0
        self.running = True




def main(args=None):
    rclpy.init(args=args)
    node = Tester()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

