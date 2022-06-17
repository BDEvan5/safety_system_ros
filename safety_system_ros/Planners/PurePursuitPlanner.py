
from safety_system_ros.utils.pp_utils import *


import numpy as np
# from RacingRewards.Utils.utils import init_file_struct
from numba import njit

class PurePursuitPlanner:
    def __init__(self, conf, map_name):
        self.name = "PurePursuitPlanner"
        

        self.trajectory = Trajectory(map_name)
        # self.trajectory.show_pts()

        self.lookahead = conf.lookahead
        self.v_min_plan = conf.v_min_plan
        self.wheelbase =  conf.l_f + conf.l_r
        self.max_steer = conf.max_steer
        self.vehicle_speed = conf.vehicle_speed


    def plan(self, obs):
        state = obs['state']
        position = state[0:2]
        theta = state[2]
        lookahead_point = self.trajectory.get_current_waypoint(position, self.lookahead)

        # TODO: this should only be for simulation
        # if state[3] < self.v_min_plan:
        #     return np.array([0.0, self.vehicle_speed])

        speed, steering_angle = get_actuation(theta, lookahead_point, position, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        speed = self.vehicle_speed
        action = np.array([steering_angle, speed])
        # action = np.array([steering_angle, self.vehicle_speed])

        return action

