from abc import abstractmethod
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from safety_system_ros.Supervisor import Supervisor

import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from safety_system_ros.utils.Dynamics import *
from safety_system_ros.utils.util_functions import *
from copy import copy
from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner


from transforms3d import euler



class SuperP(Node):
    def __init__(self):
        super().__init__('super_p')
        self.conf = load_conf("config_file") 
        
        # abstract variables
        self.planner = None
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
        self.cmd_timer = self.create_timer(self.conf.simulation_time, self.drive_callback)

        odom_topic = "pf/pose/odom"
        self.odom_subscriber = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        self.ego_reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose', 10)

        self.ns_pub = self.create_publisher(
            PoseStamped,
            '/ns_pose', 10)

        self.declare_parameter('map_name')
        map_name = self.get_parameter('map_name').value
        self.get_logger().info(f"Map param: {map_name}")

        self.planner = PurePursuitPlanner(self.conf, map_name)
        self.get_logger().info(self.planner.name)
        self.get_logger().info(f"Pure pursuit enabled")

        self.supervisor = Supervisor(self.conf, map_name)
        self.get_logger().info("Supervision always enabled")

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

        self.scan = scan

    def lap_done(self):
        self.current_lap_time = time.time() - self.lap_start_time
        self.get_logger().info(f"Run {self.lap_count} Complete in time: {self.current_lap_time}")
        self.lap_complete_callback()

        self.lap_count += 1

        self.save_data_callback()

        if self.logger: self.logger.reset_logging()

        self.current_lap_time = 0.0
        self.num_toggles = 0
        self.near_start = True
        self.toggle_list = 0
        self.lap_start_time = time.time()

    def save_data_callback(self):
        pass

    def drive_callback(self):
        if self.check_lap_done(self.position):
            self.lap_done()
        
        observation = self.build_observation()

        action = self.calculate_action(observation)

        self.send_drive_message(action)

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
        elif not closes and self.near_start:
            self.near_start = False
            self.toggle_list += 1
            # print(self.toggle_list)
        self.lap_counts = self.toggle_list // 2
        
        done = self.toggle_list >= 2
        
        return done


    def calculate_action(self, observation):
        state = observation['state']
        safe = self.supervisor.check_kernel_state(state)
        if safe:
            action = self.planner.plan(observation['state'])
            if action[1] > 0.1:  action[1] = 1
            self.get_logger().info(f"State unsafe -->  PP: {action}")
            return action

        observation['state'][3] = self.conf.vehicle_speed 
        rand_steer = (np.random.random() - 0.5) * 0.3
        action = np.array([rand_steer, self.conf.vehicle_speed]) # select action here.
        # action = np.array([0, self.conf.vehicle_speed]) # select action here.

        safe, next_state = self.supervisor.check_init_action(state, action)
        self.send_pose_msg(next_state)
        action = action.astype(float)
        if not safe:
            if action[1] > 0.1:  action[1] = 1
            self.get_logger().info(f"Init: {action}")
            return action
        else:
            action = self.planner.plan(state)
            if action[1] > 0.1:  action[1] = 1
            self.get_logger().info(f"PP: {action}")
            return action



    def lap_complete_callback(self):
        print(f"Lap complee: {self.current_lap_time}")
        if self.supervision:
            print(f"Interventions: {self.supervisor.interventions}")

    def send_pose_msg(self, ns):
        # ns = self.supervisor.next_state.copy()

        msg = PoseStamped() 

        ts = self.get_clock().now().to_msg()
        msg.header.stamp = ts
        msg.header.frame_id = 'map'

        pose_quat = euler.euler2quat(0., 0., ns[2], axes='sxyz')
        msg.pose.orientation.x = pose_quat[1]
        msg.pose.orientation.y = pose_quat[2]
        msg.pose.orientation.z = pose_quat[3]
        msg.pose.orientation.w = pose_quat[0]
        msg.pose.position.x = ns[0]
        msg.pose.position.y = ns[1]

        self.ns_pub.publish(msg)





def main(args=None):
    rclpy.init(args=args)
    node = SuperP()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


