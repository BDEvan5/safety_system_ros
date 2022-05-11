import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from safety_system_ros.utils import *
# from safety_system_ros.PurePursuitPlanner import PurePursuitPlanner 

from safety_system_ros.Dynamics import *
from safety_system_ros.PurePursuitPlanner import PurePursuitPlanner
from copy import copy



class RandomPlanner:
    def __init__(self, conf, name="RandoPlanner"):
        self.d_max = conf.max_steer # radians  
        self.name = name
        self.speed = conf.vehicle_speed

        # path = os.getcwd() + f"/{conf.vehicle_path}" + self.name 
        # init_file_struct(path)
        # self.path = path
        # np.random.seed(conf.random_seed)

    def plan(self, pos, th):
        steering = np.random.uniform(-self.d_max, self.d_max)
        return np.array([steering, self.speed])


class Supervisor(Node):
    def __init__(self):
        super().__init__('supervisor')
        
        self.d_max = 0.4
        conf = load_conf("config_file")
        self.kernel = TrackKernel(conf, True)

        self.safe_history = SafetyHistory()
        self.intervene = False

        self.time_step = conf.lookahead_time_step

        # self.planner = PurePursuitPlanner()
        self.planner = RandomPlanner(conf)

        self.m = Modes(conf)
        self.interventions = 0


        self.position = np.array([0, 0])
        self.velocity = 0
        self.theta = 0
        self.state = np.zeros(5)

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_timer = self.create_timer(0.05, self.send_cmd_msg)

        self.odom_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.current_drive_sub = self.create_subscription(AckermannDrive, 'ego_racecar/current_drive', self.current_drive_callback, 10)

    def current_drive_callback(self, msg):
        self.steering_angle = msg.steering_angle

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.position = np.array([position.x, position.y])
        self.velocity = msg.twist.twist.linear.x

        x, y, z = quaternion_to_euler_angle(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        theta = z * np.pi / 180
        self.theta = copy(theta)

    def send_cmd_msg(self):
        action = self.planner.plan(self.position, self.theta)
        safe_action = self.supervise(action) 

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = safe_action[1]
        drive_msg.drive.steering_angle = safe_action[0]
        self.drive_publisher.publish(drive_msg)

    def supervise(self, init_action):
        state = np.array([self.position[0], self.position[1], self.theta, self.velocity, self.steering_angle]).copy()
        safe, next_state = self.check_init_action(state, init_action)
        if safe:
            self.safe_history.add_locations(init_action, init_action)
            return init_action
        # self.get_logger().info(f"Unsafe Action: {init_action}")

        self.interventions += 1
        valids = self.simulate_and_classify(state)
        if not valids.any():
            print(f"No Valid options -> State: {state}")
            return init_action
        
        action, idx = modify_mode(valids, self.m.qs)
        self.safe_history.add_locations(init_action, action)

        return action

    def check_init_action(self, state, init_action):
        self.kernel.plot_state(state)

        # next_state = run_dynamics_update(state, init_action, self.time_step/2)
        # safe = check_kernel_state(next_state, self.kernel.kernel, self.kernel.origin, self.kernel.resolution, self.kernel.phi_range, self.m.qs)
        # if not safe:
        #     return safe, next_state

        next_state = run_dynamics_update(state, init_action, self.time_step)
        safe = check_kernel_state(next_state, self.kernel.kernel, self.kernel.origin, self.kernel.resolution, self.kernel.phi_range, self.m.qs)
        
        return safe, next_state

    def simulate_and_classify(self, state):
        valids = np.ones(len(self.m.qs))
        for i in range(len(self.m.qs)):
            next_state = run_dynamics_update(state, self.m.qs[i], self.time_step)
            valids[i] = check_kernel_state(next_state, self.kernel.kernel, self.kernel.origin, self.kernel.resolution, self.kernel.phi_range, self.m.qs)
            # self.kernel.plot_state(next_state)

        return valids




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




@njit(cache=True)
def modify_mode(valid_window, qs):
    """ 
    modifies the action for obstacle avoidance only, it doesn't check the dynamic limits here.
    """
    assert valid_window.any() == 1, "No valid actions:check modify_mode method"

    idx_search = int((len(qs)-1)/2)
    if valid_window[idx_search]:
        return qs[idx_search], idx_search

    d_search_size = int((len(qs)-1)/2)
    for dind in range(d_search_size+1): 
        p_d = int(idx_search+dind)
        if valid_window[p_d]:
            return qs[p_d], p_d
        n_d = int(idx_search-dind-1)
        if valid_window[n_d]:
            return qs[n_d], n_d
        
@njit(cache=True)
def check_state_modes(v, d):
    b = 0.523
    g = 9.81
    l_d = 0.329
    if abs(d) < 0.06:
        return True # safe because steering is small
    friction_v = np.sqrt(b*g*l_d/np.tan(abs(d))) *1.1 # nice for the maths, but a bit wrong for actual friction
    if friction_v > v:
        return True # this is allowed mode
    return False # this is not allowed mode: the friction is too high

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

        d_min, di = 1000, None
        for i in range(len(qs)):
            d_dis = abs(qs[i, 0] - state[4])
            if d_dis < d_min:
                d_min, di = d_dis, i
        d_ind = min(max(0, int(round(di))), qs.shape[0]-1)
        mode = int(d_ind)
        
        if kernel[x_ind, y_ind, theta_ind, mode] != 0:
            return False # unsfae state
        return True # safe state



class TrackKernel:
    def __init__(self, sim_conf, plotting=False):
        map_name = "columbia_small"
        kernel_name = f"/home/benjy/sim_ws/src/safety_system_ros/Data/Kernels/Kernel_viab_{map_name}.npy"
        self.kernel = np.load(kernel_name)

        self.plotting = plotting
        self.m = Modes(sim_conf)
        self.resolution = sim_conf.n_dx
        self.phi_range = sim_conf.phi_range
        self.max_steer = sim_conf.max_steer
        self.n_modes = self.m.n_modes

        
        file_name = f'/home/benjy/sim_ws/src/safety_system_ros/map_data/' + sim_conf.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = np.array(yaml_file['origin'])

    def check_state(self, state=[0, 0, 0, 0, 0]):
        i, j, k, m = self.get_indices(state)

        if self.plotting:
            self.plot_kernel_point(i, j, k, m)
        if self.kernel[i, j, k, m] != 0:
            return False # unsfae state
        return True # safe state

    def plot_kernel_point(self, i, j, k, m):
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

        phi = state[2]
        if phi >= self.phi_range/2:
            phi = phi - self.phi_range
        elif phi < -self.phi_range/2:
            phi = phi + self.phi_range
        theta_ind = int(round((phi + self.phi_range/2) / self.phi_range * (self.kernel.shape[2]-1)))
        mode = self.m.get_mode_id(state[4])

        return x_ind, y_ind, theta_ind, mode

    def plot_state(self, state):
        i, j, k, m = self.get_indices(state)
        self.plot_kernel_point(i, j, k, m)


class SafetyHistory:
    def __init__(self):
        self.planned_actions = []
        self.safe_actions = []

    def add_locations(self, planned_action, safe_action=None):
        self.planned_actions.append(planned_action)
        if safe_action is None:
            self.safe_actions.append(planned_action)
        else:
            self.safe_actions.append(safe_action)

    def plot_safe_history(self):
        planned = np.array(self.planned_actions)
        safe = np.array(self.safe_actions)
        plt.figure(5)
        plt.clf()
        plt.title("Safe History: steering")
        plt.plot(planned[:, 0], color='blue')
        plt.plot(safe[:, 0], '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        plt.figure(6)
        plt.clf()
        plt.title("Safe History: velocity")
        plt.plot(planned[:, 1], color='blue')
        plt.plot(safe[:, 1], '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        self.planned_actions = []
        self.safe_actions = []

    def save_safe_history(self, path, name):
        self.plot_safe_history()

        plt.figure(5)
        plt.savefig(f"{path}/{name}_steer_actions.png")

        plt.figure(6)
        plt.savefig(f"{path}/{name}_velocity_actions.png")

        data = []
        for i in range(len(self.planned_actions)):
            data.append([i, self.planned_actions[i], self.safe_actions[i]])
        full_name = path + f'/{name}_training_data.csv'
        with open(full_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)


        self.planned_actions = []
        self.safe_actions = []


def main(args=None):
    rclpy.init(args=args)
    node = Supervisor()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


