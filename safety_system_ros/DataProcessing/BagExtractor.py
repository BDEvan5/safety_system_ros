from turtle import pos
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan


from safety_system_ros.utils.util_functions import *

import numpy as np

directory = "/home/benjy/sim_ws/src/safety_system_ros/"


import time

class BagExtractor(Node):
    def __init__(self):
        super().__init__('bag_extractor')

        self.declare_parameter('bag_name')
        self.bag_name = self.get_parameter('bag_name').value

        self.path = directory + "Data/BagData/" + self.bag_name + "/"
        init_file_struct(self.path)
        print(f"Initated file structure: {self.bag_name}")
        self.odom_log_file = self.path + self.bag_name + "_odom.csv"
        self.action_log_file = self.path + self.bag_name + "_actions.csv"

        self.odom_sub = self.create_subscription(Odometry, 'pf/pose/odom', self.odom_callback, 10)

        self.action_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.action_callback, 10)

        self.started = False
        self.timer_called = False
        self.timer = self.create_timer(0.2, self.timer_callback)

        time.sleep(0.5)
        self.get_logger().info(f"Node running")

    def action_callback(self, msg):
        action = np.zeros(2)
        action[0] = msg.drive.speed
        action[1] = msg.drive.steering_angle
        time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.add_action_line(action, time_stamp)

    def timer_callback(self):
        if self.started and self.timer_called:
            self.get_logger().info(f"Finished saving: Destroy!!!")
            self.destroy_node()
        self.timer_called = True

    def odom_callback(self, msg):
        self.timer_called = False
        self.started = True
        position = msg.pose.pose.position
        position = np.array([position.x, position.y])
        velocity = msg.twist.twist.linear.x

        theta = orientation_to_angle(msg.pose.pose.orientation)
        # print(f"Odom: {position}, {theta}, {velocity }")
        time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        # if velocity > 0:
        self.add_data_line(position, velocity, theta, time_stamp)

    def add_data_line(self, position, velocity, theta, time_stamp):
        with open(self.odom_log_file, "a") as f:
            data = f"{position[0]}, {position[1]}, {theta}, {velocity}, {time_stamp}"
            f.write(data + "\n")

    def add_action_line(self, action, time_stamp):
        with open(self.action_log_file, "a") as f:
            data = f"{action[0]}, {action[1]}, {time_stamp}"
            f.write(data + "\n")

def main():
    """Main function for ros node"""
    rclpy.init()
    node = BagExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
