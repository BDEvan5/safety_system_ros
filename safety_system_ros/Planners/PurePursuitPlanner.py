
from safety_system_ros.utils.pp_utils import *


import numpy as np
# from RacingRewards.Utils.utils import init_file_struct
from numba import njit

class PurePursuitPlanner:
    def __init__(self, conf):

        self.trajectory = Trajectory(conf.map_name)
        # self.trajectory.show_pts()

        self.lookahead = 1
        self.v_min_plan = 1
        self.wheelbase =  0.33
        self.max_steer = 0.4
        self.vehicle_speed = 2.0

    def plan(self, obs):
        state = obs['state']
        position = state[0:2]
        theta = state[2]
        lookahead_point = self.trajectory.get_current_waypoint(position, self.lookahead)

        speed, steering_angle = get_actuation(theta, lookahead_point, position, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        return np.array([steering_angle, self.vehicle_speed])

