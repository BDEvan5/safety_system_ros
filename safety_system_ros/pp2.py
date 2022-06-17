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

from safety_system_ros.Supervisor import Supervisor
from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner



class PP2(Node):
    def __init__(self):
        super().__init__('pp2')
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

        self.logger = None

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        # self.cmd_timer = self.create_timer(self.conf.simulation_time, self.drive_callback)

        odom_topic = "pf/pose/odom"
        # odom_topic = "ego_racecar/odom"
        self.odom_subscriber = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)


        self.ego_reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose', 10)


        self.declare_parameter('n_laps')
        self.declare_parameter('supervision')
        self.declare_parameter('map_name')

        map_name = self.get_parameter('map_name').value

        self.planner = PurePursuitPlanner(self.conf, map_name)
        self.get_logger().info(self.planner.name)

        self.supervision = self.get_parameter('supervision').value
        if self.supervision:
            self.get_logger().info("Supervision enabled")
            self.supervisor = Supervisor(self.conf, map_name)

        self.n_laps = self.get_parameter('n_laps').value
        self.get_logger().info(f"Number of laps to run: {self.n_laps}")

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.position = np.array([position.x, position.y])
        self.velocity = msg.twist.twist.linear.x

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        theta = z * np.pi / 180
        self.theta = copy(theta)

        # self.drive_callback()

    def drive_callback(self):
        state = np.array([self.position[0], self.position[1], self.theta, self.velocity, 0])

        action = self.planner.plan(state) 
        if self.supervision: 
            new_action =  self.supervisor.supervise(state, action)
            if new_action[0] == action[0]:
                self.get_logger().info(f"Action: {action}")
            else:
                self.get_logger().info(f"Action: {action} :: NewAct: {new_action}")

            action = new_action
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(action[1])
        drive_msg.drive.steering_angle = float(action[0])
        self.drive_publisher.publish(drive_msg)


    def ego_reset(self):
        msg = PoseWithCovarianceStamped() 

        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        self.ego_reset_pub.publish(msg)

        self.get_logger().info("Finished Resetting")


    def lap_complete_callback(self):
        print(f"Lap complee: {self.current_lap_time}")
        if self.supervision:
            print(f"Interventions: {self.supervisor.interventions}")


def main(args=None):
    rclpy.init(args=args)
    node = PP2()
    rclpy.spin(node)

if __name__ == '__main__':
    main()