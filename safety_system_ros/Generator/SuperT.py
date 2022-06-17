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


class Modes:
    def __init__(self, conf) -> None:
        self.time_step = conf.kernel_time_step
        self.nq_steer = conf.nq_steer
        self.max_steer = conf.max_steer
        vehicle_speed = conf.vehicle_speed

        ds = np.linspace(-self.max_steer, self.max_steer, self.nq_steer)
        vs = vehicle_speed * np.ones_like(ds)
        self.qs = np.stack((ds, vs), axis=1)

        self.n_modes = len(self.qs)

    def get_mode_id(self, delta):
        d_ind = np.argmin(np.abs(self.qs[:, 0] - delta))
        
        return int(d_ind)

    def action2mode(self, action):
        id = self.get_mode_id(action[0])
        return self.qs[id]

    def __len__(self): return self.n_modes
    

class SuperT(Node):
    def __init__(self):
        super().__init__('super_t')
        self.conf = load_conf("config_file") 
        
        # abstract variables
        self.supervisor = None

        # current vehicle state
        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0
        self.steering_angle = 0.0
        self.scan = None # move to param file

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_timer = self.create_timer(self.conf.simulation_time, self.drive_callback)

        odom_topic = "pf/pose/odom"
        self.odom_subscriber = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.ego_reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose', 10)

        self.declare_parameter('map_name')
        map_name = self.get_parameter('map_name').value

        kernel_name = self.conf.directory + f"Data/Kernels/Kernel_transform_{map_name}.npy"
        self.m = Modes(self.conf)

        self.kernel = np.load(kernel_name)

        self.resolution = self.conf.n_dx
        self.phi_range = self.conf.phi_range
        self.max_steer = self.conf.max_steer
        self.n_modes = self.m.n_modes
        
        file_name = self.conf.directory + f'map_data/' + map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = np.array(yaml_file['origin'])


    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.position = np.array([position.x, position.y])
        self.velocity = msg.twist.twist.linear.x

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        theta = z * np.pi / 180
        self.theta = copy(theta)

    def drive_callback(self):
        state = np.array([self.position[0], self.position[1], self.theta, self.velocity, 0])
        # self.get_logger().info(f"State: {state}")

        action = self.get_kernel_actions(state)
        self.get_logger().info(f"State: {state} --> Action: {action}")

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(action[1])
        drive_msg.drive.steering_angle = float(action[0])
        self.drive_publisher.publish(drive_msg)

    def get_kernel_actions(self, state):
        actions = check_kernel_actions(state, self.kernel, self.resolution, self.phi_range)

        n_search = 4
        if not actions[n_search]:
            return self.m.qs[n_search]
        for i in range(4):
            if not actions[i+n_search]:
                return self.m.qs[i+n_search]
            if not actions[i-n_search]:
                return self.m.qs[i-n_search]
        print(f"No options")
        return np.array([0, 0])

        


@njit(cache=True)
def check_kernel_actions(state, kernel, origin, resolution, phi_range):
        x_ind = min(max(0, int(round((state[0]-origin[0])*resolution))), kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1]-origin[1])*resolution))), kernel.shape[1]-1)

        phi = state[2]
        if phi >= phi_range/2:
            phi = phi - phi_range
        elif phi < -phi_range/2:
            phi = phi + phi_range
        theta_ind = int(round((phi + phi_range/2) / phi_range * (kernel.shape[2]-1)))

        return kernel[x_ind, y_ind, theta_ind, :]



def main(args=None):
    rclpy.init(args=args)
    node = SuperT()
    rclpy.spin(node)

if __name__ == '__main__':
    main()