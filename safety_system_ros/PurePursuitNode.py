from safety_system_ros.DriveNode import DriveNode

import rclpy
import time
import numpy as np

from safety_system_ros.utils.util_functions import *
from safety_system_ros.utils.pp_utils import *

class PurePursuitNode(DriveNode):
    def __init__(self):
        super().__init__("pure_pursuit")

        self.declare_parameter('n_laps')
        self.declare_parameter('map_name')
        map_name = self.get_parameter('map_name').value

        self.trajectory = Trajectory(map_name)

        self.lookahead = self.conf.lookahead
        self.v_min_plan = self.conf.v_min_plan
        self.wheelbase =  self.conf.l_f + self.conf.l_r
        self.max_steer = self.conf.max_steer
        self.vehicle_speed = self.conf.vehicle_speed

        self.n_laps = self.get_parameter('n_laps').value
        self.get_logger().info(f"Number of laps to run: {self.n_laps}")

    def calculate_action(self, observation):
        state = observation['state']
        position = state[0:2]
        theta = state[2]
        lookahead_point = self.trajectory.get_current_waypoint(position, self.lookahead)

        speed, steering_angle = get_actuation(theta, lookahead_point, position, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # action = np.array([steering_angle, speed])
        action = np.array([steering_angle, self.vehicle_speed])

        return action

    def lap_complete_callback(self):
        print(f"Lap complee: {self.current_lap_time}")


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


