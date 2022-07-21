from safety_system_ros.Supervisor import Supervisor
from safety_system_ros.DriveNode import DriveNode

import rclpy
import time
import numpy as np

from safety_system_ros.utils.util_functions import *
from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner
from safety_system_ros.utils.Dynamics import *


class SingleMode:
    def __init__(self, conf) -> None:
        self.qs = np.array([[0, conf.vehicle_speed]])
        self.n_modes = 1

    def get_mode_id(self, delta):
        return 0

    def action2mode(self, action):
        return self.qs[0]

    def __len__(self): return self.n_modes
   
    
class Supervisor:
    def __init__(self, conf, map_name):
        self.time_step = conf.lookahead_time_step

        kernel_name = conf.directory + f"Data/Kernels/Kernel_filter_{map_name}.npy"
        self.m = SingleMode(conf)
        self.kernel = np.load(kernel_name)

        self.resolution = conf.n_dx
        self.phi_range = conf.phi_range
        self.max_steer = conf.max_steer
        self.n_modes = self.m.n_modes

        file_name = conf.directory + f'map_data/' + map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = np.array(yaml_file['origin'])


    def check_init_action(self, state, init_action):
        next_state = run_dynamics_update(state, init_action, self.time_step/2)
        safe = check_kernel_state(next_state, self.kernel, self.origin, self.resolution, self.phi_range, self.m.qs)
        if not safe:
            return safe, next_state

        next_state = run_dynamics_update(state, init_action, self.time_step)
        safe = check_kernel_state(next_state, self.kernel, self.origin, self.resolution, self.phi_range, self.m.qs)
        
        return safe, next_state

    def check_kernel_state(self, state):
        self.plot_kernel_point(state)
        safe = check_kernel_state(state, self.kernel, self.origin, self.resolution, self.phi_range, self.m.qs)
        return safe

    def plot_kernel_point(self, state):
        i, j, k, m = self.get_indices(state)
        plt.figure(6)
        plt.clf()
        plt.title(f"Kernel inds: {i}, {j}, {k}, {m}: {self.m.qs[m]}")
        img = self.kernel[:, :, k, m].T 
        plt.imshow(img, origin='lower')
        plt.plot(i, j, 'x', markersize=20, color='red')
        plt.pause(0.0001)


    def get_indices(self, state):
        x_ind = min(max(0, int(round((state[0]-self.origin[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1]-self.origin[1])*self.resolution))), self.kernel.shape[1]-1)

        phi = limit_phi(state[2])
        theta_ind = int(round((phi + self.phi_range/2) / self.phi_range * (self.kernel.shape[2]-1)))
        mode = 0
        return x_ind, y_ind, theta_ind, mode
     
@njit(cache=True) 
def check_kernel_state(state, kernel, origin, resolution, phi_range, qs):
        x_ind = min(max(0, int(round((state[0]-origin[0])*resolution))), kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1]-origin[1])*resolution))), kernel.shape[1]-1)

        phi = state[2]
        if phi >= phi_range/2:
            phi = phi - phi_range
        elif phi < -phi_range/2:
            phi = phi + phi_range
        theta_ind = int(round((phi + phi_range/2) / phi_range * (kernel.shape[2]-1)))

        mode = 0
        
        if kernel[x_ind, y_ind, theta_ind, mode] != 0:
            return False # unsfae state
        return True # safe state

class SuperRando(DriveNode):
    def __init__(self):
        super().__init__('rando_plan')

        self.declare_parameter('n_laps')
        self.declare_parameter('map_name')

        map_name = self.get_parameter('map_name').value
        self.n_laps = self.get_parameter('n_laps').value

        self.get_logger().info(f"Number of random testing laps: {self.n_laps}")

        self.vehicle_speed = self.conf.vehicle_speed
        self.d_max = self.conf.max_steer
        
        file_name = self.conf.directory + f'map_data/' + map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = np.array(yaml_file['origin'])

        self.pp_planner = PurePursuitPlanner(self.conf, map_name)
        self.supervisor = Supervisor(self.conf, map_name)

        self.intervenes = 0

    def calculate_action(self, observation):
        # return action
        state = observation['state']
        safe = self.supervisor.check_kernel_state(state)
        if not safe:
            action = self.pp_planner.plan(observation)
            self.get_logger().info(f"State unsafe: ")
            if action[1] > 0.1:  action[1] = self.vehicle_speed
            return action


        # select random action
        # steering = np.clip(np.random.normal(0, 0.1), -self.d_max, self.d_max)
        # init_action = np.array([steering, self.vehicle_speed])
        init_action = self.pp_planner.plan(observation)

        state = observation['state']

        safe, next_state = self.supervisor.check_init_action(state, init_action)
        if safe:
            if init_action[1] > 0.1:  init_action[1] = self.vehicle_speed
            return init_action
        else:
            self.get_logger().info(f"Prediction Unsafe: ")
            action = self.pp_planner.plan(observation)
            self.intervenes += 1
            if action[1] > 0.1:  action[1] = self.vehicle_speed
            return action


    def lap_complete_callback(self):
        print(f"Lap complee: {self.current_lap_time}")
        print(f"Interventions: {self.intervenes}")



def main(args=None):
    rclpy.init(args=args)
    node = SuperRando()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


