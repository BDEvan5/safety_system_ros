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



class DriveNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.conf = load_conf("config_file") 
        
        # abstract variables
        self.planner = None
        self.supervision = False 
        self.supervisor = None

        # current vehicle state
        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0
        self.steering_angle = 0.0
        self.scan = None # move to param file

        self.lap_counts = 0
        self.toggle_list = 0
        self.near_start = True
        self.lap_start_time = time.time()
        self.current_lap_time = 0.0
        self.running = False

        self.lap_count = 0 
        self.n_laps = None
        self.lap_times = []

        self.logger = None

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_timer = self.create_timer(self.conf.simulation_time, self.drive_callback)

        self.odom_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.current_drive_sub = self.create_subscription(AckermannDrive, 'ego_racecar/current_drive', self.current_drive_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        self.ego_reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose', 10)

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
        self.scan = np.array(msg.ranges)

    def lap_done(self):
        self.current_lap_time = time.time() - self.lap_start_time
        self.get_logger().info(f"Run {self.lap_count} Complete in time: {self.current_lap_time}")
        self.lap_times.append(self.current_lap_time)
        self.lap_complete_callback()

        self.lap_count += 1

        if self.lap_count == self.n_laps:
            self.running = False
            self.save_data_callback()
            self.ego_reset()
            self.destroy_node()

        if self.logger: self.logger.reset_logging()

        self.current_lap_time = 0.0
        self.num_toggles = 0
        self.near_start = True
        self.toggle_list = 0
        self.lap_start_time = time.time()

    def drive_callback(self):
        if not self.running:
            return

        if self.check_lap_done(self.position):
            self.lap_done()
        
        observation = self.build_observation()

        action = self.calculate_action(observation)

        self.send_drive_message(action)

    @abstractmethod
    def calculate_action(self, observation):
        """
            Use the observation to calculate an action that is returned
        """
        raise NotImplementedError

    @abstractmethod
    def save_data_callback(self):
        # self.get_logger().info("No extra data to save")
        # with open("/home/benjy/sim_ws/src/safety_system_ros/Data/DriveNodeLapLog.csv") as file:
        lap_times = np.array(self.lap_times)
        np.save("/home/benjy/sim_ws/src/safety_system_ros/Data/DriveNodeLapLog.npy", lap_times)
        self.get_logger().info(f"Saved lap times: {lap_times}")


    @abstractmethod
    def lap_complete_callback(self):
        pass

    def send_drive_message(self, action):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(action[1])
        drive_msg.drive.steering_angle = float(action[0])
        self.drive_publisher.publish(drive_msg)

    def build_observation(self):
        """
        Observation:
            scan: LiDAR scan 
            state: [pose_x, pose_y, theta, velocity, steering angle]
            reward: 0 - created here to store the reward later
        
        """
        observation = {}
        observation["scan"] = self.scan
        if observation["scan"] is None: observation["scan"] = np.zeros(1080)

        state = np.array([self.position[0], self.position[1], self.theta, self.velocity, self.steering_angle])
        observation['state'] = state
        observation['reward'] = 0.0

        return observation

    def check_lap_done(self, position):
        start_x = 0
        start_y = 0 
        start_theta = 0
        start_rot = np.array([[np.cos(-start_theta), -np.sin(-start_theta)], [np.sin(-start_theta), np.cos(-start_theta)]])

        poses_x = np.array(position[0])-start_x
        poses_y = np.array(position[1])-start_y
        delta_pt = np.dot(start_rot, np.stack((poses_x, poses_y), axis=0))

        dist2 = delta_pt[0]**2 + delta_pt[1]**2
        closes = dist2 <= 1
        if closes and not self.near_start:
            self.near_start = True
            self.toggle_list += 1
            self.get_logger().info(f"Near start true: {position}")
        elif not closes and self.near_start:
            self.near_start = False
            self.toggle_list += 1
            self.get_logger().info(f"Near start false: {position}")
            # print(self.toggle_list)
        self.lap_counts = self.toggle_list // 2
        
        done = self.toggle_list >= 2
        
        return done

    def ego_reset(self):
        msg = PoseWithCovarianceStamped() 

        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        # msg.pose.pose.orientation.z = -0.707
        # msg.pose.pose.orientation.w = 0.707
        # msg.pose.pose.orientation.x = 0.0
        # msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 1.0
        msg.pose.pose.orientation.w = 0.0

        self.ego_reset_pub.publish(msg)

        self.get_logger().info("Finished Resetting: angle 180")

    def run_lap(self):
        time.sleep(0.1)
        self.ego_reset()
        time.sleep(0.1)

        self.current_lap_time = 0.0
        self.running = True


if __name__ == '__main__':
    pass

