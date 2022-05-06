import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from matplotlib import pyplot as plt


import numpy as np

from pp_utils import *
from utils import quaternion_to_euler_angle

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        self.trajectory = Trajectory('levine_blocked')

        self.lookahead = 1
        self.v_min_plan = 1
        self.wheelbase =  0.33
        self.max_steer = 0.4
        self.vehicle_speed = 2

        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_timer = self.create_timer(0.1, self.send_cmd_msg)

        self.odom_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.position = np.array([position.x, position.y])
        self.velocity = msg.twist.twist.linear.x

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        self.theta = z * np.pi / 180

    def send_cmd_msg(self):
        lookahead_point = self.trajectory.get_current_waypoint(self.position, self.lookahead)

        # plt.figure(3)
        # plt.plot(self.position[0], self.position[1], 'ro')
        # plt.plot(lookahead_point[0], lookahead_point[1], 'bo')
        # plt.title(f"Theta: {self.theta}")
        # plt.pause(0.00001)

        speed, steering_angle = get_actuation(self.theta, lookahead_point, self.position, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        speed = 2.0

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_publisher.publish(drive_msg)

        # self.get_logger().info(f"Steering angle: {steering_angle}")
        




def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)

if __name__ == '__main__':
    main()



