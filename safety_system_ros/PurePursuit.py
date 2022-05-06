import rclpy
from geometry_msgs.msg import PoseStamped, Twist, Pose, PointStamped
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from matplotlib import pyplot as plt


import numpy as np
from numba import njit
from pp_utils import *
import csv
import os

import math

def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        self.trajectory = Trajectory('levine_blocked')
        self.trajectory.show_pts()

        self.lookahead = 4
        self.v_min_plan = 1
        self.wheelbase =  0.33
        self.max_steer = 0.4
        self.vehicle_speed = 2

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        # self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_timer = self.create_timer(0.1, self.send_cmd_msg)

        self.odom_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0

        self.lookahead_pub = self.create_publisher(PointStamped, '/lookahead', 10)

        drive_msg = AckermannDriveStamped()
        
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_publisher.publish(drive_msg)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        twist = msg.twist.twist.linear
        # self.get_logger().info(f"Position: {position}")
    
        self.position = np.array([position.x, position.y])
        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        # self.get_logger().info(f"Orientation: {x}, {y}, {z}")
        self.theta = z
        self.velocity = msg.twist.twist.linear.x
        # self.get_logger().info(f"Velocity: {self.velocity}")

    def send_cmd_msg(self):
        # if self.velocity < self.v_min_plan:
            
        lookahead_point = self.trajectory.get_current_waypoint(self.position, self.lookahead)

        plt.figure(3)
        plt.plot(self.position[0], self.position[1], 'ro')
        plt.plot(lookahead_point[0], lookahead_point[1], 'bo')
        plt.title(f"Theta: {self.theta}")
        plt.pause(0.00001)

        lookahead_pose = PointStamped()
        # lookahead_pose.header.stamp = self.now()
        # lookahead_pose.header.frame_id = 'map'
        lookahead_pose.point.x = lookahead_point[0]
        lookahead_pose.point.y = lookahead_point[1]
        self.lookahead_pub.publish(lookahead_pose)

        speed, steering_angle, w_y = get_actuation(self.theta, lookahead_point, self.position, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        plt.figure(4)
        plt.clf()
        plt.plot(steering_angle*10, -1, 'go')
        plt.plot(w_y, 0, 'ro')
        plt.xlim([-4, 4])
        plt.pause(0.00001)

        speed = 0.5

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_publisher.publish(drive_msg)

        self.get_logger().info(f"Steering angle: {steering_angle}")
        




def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)

if __name__ == '__main__':
    main()



