from turtle import pos
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from safety_system_ros.utils.util_functions import *

import numpy as np

directory = "/home/benjy/sim_ws/src/safety_system_ros/"

# bag_name = "e_1_test"
# bag_name = "e_2_test"
# bag_name = "e_2_test2"
# bag_name = "e_3_test"
# bag_name = "e_4_test"
# bag_name = "e_5_test"
# bag_name = "e_2_train"
# bag_name = "LevineT_1_bag"
# bag_name = "LevineT_2"
bag_name = "PP_1"
# bag_name = "pp_lobby_1"
# bag_name = "sim2real_2"
# bag_name = "sim2real_3"
# bag_name = "sim2real_lobby"
# bag_name = "superP_run1working"
# bag_name = "superT_bag"
# bag_name = "supervisor_lobby_v2"
# bag_name = "t1_lobby_train"
# bag_name = "t1_lobby_v3"
# bag_name = "t2_1_random_super_test"
# bag_name = "t2_1_train"
# bag_name = "t2_1_train2"
# bag_name = "t2_11_train"
# bag_name = "test9"
# bag_name = "test_agent10"
# bag_name = "testing_4"
# bag_name = "testing_6"
# bag_name = "test_onBoardA_1"
# bag_name = "test_onBoardA_2"
# bag_name = "train_agent10"
# bag_name = "train_agent11"
# bag_name = "training5"
# bag_name = "training7"
# bag_name = "training_4"
# bag_name = "training_402"
# bag_name = "train_onB_3"
# bag_name = "train_onBActual_3"

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

        self.odom_sub = self.create_subscription(Odometry, 'pf/pose/odom', self.odom_callback, 10)

        self.started = False
        self.timer_called = False
        self.timer = self.create_timer(0.2, self.timer_callback)

        time.sleep(0.5)
        self.get_logger().info(f"Node running")

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
        
        if velocity > 0:
            self.add_data_line(position, velocity, theta)

    def add_data_line(self, position, velocity, theta):
        with open(self.odom_log_file, "a") as f:
            data = f"{position[0]}, {position[1]}, {theta}, {velocity}"
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
