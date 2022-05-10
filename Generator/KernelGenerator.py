import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from safety_system_ros.utils import load_conf
from DynamicsBuilder import build_dynamics_table


class KernelGenerator:
    def __init__(self, track_img, sim_conf):
        self.track_img = track_img
        self.sim_conf = sim_conf
        self.n_dx = int(sim_conf.n_dx)
        self.t_step = sim_conf.kernel_time_step
        self.n_phi = sim_conf.n_phi
        self.max_steer = sim_conf.max_steer 
        self.L = sim_conf.l_f + sim_conf.l_r

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, self.n_x/self.n_dx, self.n_x) 
        self.ys = np.linspace(0, self.n_y/self.n_dx, self.n_y)
        self.phis = np.linspace(-np.pi, np.pi, self.n_phi)
        
        self.n_modes = sim_conf.nq_steer 
        self.qs = np.linspace(-self.max_steer, self.max_steer, self.n_modes)

        self.o_map = np.copy(self.track_img)    
        self.fig, self.axs = plt.subplots(2, 2)

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.previous_kernel = np.copy(self.kernel)

        self.kernel[:, :, :, :] = self.track_img[:, :, None, None] * np.ones((self.n_x, self.n_y, self.n_phi, self.n_modes))
        
        self.dynamics = np.load(f"{sim_conf.dynamics_path}{sim_conf.kernel_mode}_dyns.npy")
        print(f"Dynamics Loaded: {self.dynamics.shape}")

    def get_filled_kernel(self):
        filled = np.count_nonzero(self.kernel)
        total = self.kernel.size
        print(f"Filled: {filled} / {total} -> {filled/total}")
        return filled/total

    def view_kernel_angles(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        mode_ind = int((self.n_modes-1)/2)

        self.axs[0, 0].imshow(self.kernel[:, :, 0, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel phi: {self.phis[0]}")
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel phi: {self.phis[half_phi]}")
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel phi: {self.phis[-quarter_phi]}")
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel phi: {self.phis[quarter_phi]}")

        plt.pause(0.0001)
        # plt.pause(1)

        if show:
            plt.show()

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = viability_loop(self.kernel, self.dynamics)

            self.view_kernel_angles(False)
            self.get_filled_kernel()

        return self.get_filled_kernel()



@njit
def viability_loop(kernel, dynamics):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis, l_qs = kernel.shape
    for i in range(l_xs):
        for j in range(l_ys):
            for k in range(l_phis):
                for q in range(l_qs):
                    if kernel[i, j, k, q] == 1:
                        continue 
                    kernel[i, j, k, q] = check_viable_state(i, j, k, q, dynamics, previous_kernel)

    return kernel


@njit
def check_viable_state(i, j, k, q, dynamics, previous_kernel):
    l_xs, l_ys, l_phis, n_modes = previous_kernel.shape
    for l in range(n_modes):
        safe = True
        di, dj, new_k, new_q = dynamics[k, q, l, 0, :]

        if new_q == -9223372036854775808:
            continue

        for n in range(dynamics.shape[3]): # cycle through 8 block states
            di, dj, new_k, new_q = dynamics[k, q, l, n, :]

                # return True # not safe.
            new_i = min(max(0, i + di), l_xs-1)  
            new_j = min(max(0, j + dj), l_ys-1)
            # new_k = min(max(0, k + dk), l_phis-1)

            if previous_kernel[new_i, new_j, new_k, new_q]:
                # if you hit a constraint, break
                safe = False # breached a limit.
                break # try again and look for a new action

        if safe: # there exists a valid action
            return False # it is safe

    return True # it isn't safe because I haven't found a valid action yet...


def prepare_track_img(sim_conf):
    file_name =  "/home/benjy/sim_ws/src/safety_system_ros/map_data/"  + sim_conf.map_name  + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)
        yaml_file = dict(documents.items())
    img_resolution = yaml_file['resolution']
    map_img_path =  "/home/benjy/sim_ws/src/safety_system_ros/map_data/" + yaml_file['image']

    resize = int(sim_conf.n_dx * img_resolution)

    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    map_img = map_img.astype(np.float64)
    if len(map_img.shape) == 3:
        map_img = map_img[:, :, 0]
    map_img[map_img <= 128.] = 1.
    map_img[map_img > 128.] = 0.

    img = Image.fromarray(map_img.T)
    img = img.resize((map_img.shape[0]*resize, map_img.shape[1]*resize))
    img = np.array(img)
    map_img2 = img.astype(np.float64)
    map_img2[map_img2 != 0.] = 1.

    return map_img2

@njit(cache=True)
def shrink_img(img, n_shrinkpx):
    o_img = np.copy(img)

    search = np.array([[0, 1], [1, 0], [0, -1], 
                [-1, 0], [1, 1], [1, -1], 
                [-1, 1], [-1, -1]])
    for i in range(n_shrinkpx):
        t_img = np.copy(img)
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[j, k] == 1:
                    continue
                for l in range(len(search)):
                    di, dj = search[l, :]
                    new_i = min(max(0, j + di), img.shape[0]-1)
                    new_j = min(max(0, k + dj), img.shape[1]-1)
                    if t_img[new_i, new_j] == 1:
                        img[j, k] = 1.
                        break

    print(f"Finished Shrinking")
    return o_img, img #


class VeiwKernel:
    def __init__(self, conf, track_img):
        kernel_name = f"{conf.kernel_path}Kernel_{conf.kernel_mode}_{conf.map_name}.npy"
        self.kernel = np.load(kernel_name)

        self.o_map = np.copy(track_img)    
        self.fig, self.axs = plt.subplots(2, 2)

        
        self.phis = np.linspace(-conf.phi_range/2, conf.phi_range/2, conf.n_phi)

        self.qs = np.linspace(-conf.max_steer, conf.max_steer, conf.nq_steer)
        self.view_speed_build(True)
     
    def view_speed_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        phi_ind = int(len(self.phis)/2)

        inds = np.array([3, 4, 7, 8], dtype=int)

        self.axs[0, 0].imshow(self.kernel[:, :, phi_ind, inds[0]].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel Mode: {self.qs[inds[0]]}")
        self.axs[1, 0].imshow(self.kernel[:, :, phi_ind, inds[1]].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel Mode: {self.qs[inds[1]]}")
        self.axs[0, 1].imshow(self.kernel[:, :, phi_ind, inds[2]].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel Mode: {self.qs[inds[2]]}")

        self.axs[1, 1].imshow(self.kernel[:, :, phi_ind, inds[3]].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel Mode: {self.qs[inds[3]]}")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()
        
    def make_picture(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)


        self.axs[0, 0].set(xticks=[])
        self.axs[0, 0].set(yticks=[])
        self.axs[1, 0].set(xticks=[])
        self.axs[1, 0].set(yticks=[])
        self.axs[0, 1].set(xticks=[])
        self.axs[0, 1].set(yticks=[])
        self.axs[1, 1].set(xticks=[])
        self.axs[1, 1].set(yticks=[])

        self.axs[0, 0].imshow(self.kernel[:, :, 0].T + self.o_map.T, origin='lower')
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi].T + self.o_map.T, origin='lower')
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi].T + self.o_map.T, origin='lower')
        
        plt.pause(0.0001)
        plt.pause(1)
        plt.savefig(f"{self.sim_conf.kernel_path}Kernel_build_{self.sim_conf.kernel_mode}.svg")

        if show:
            plt.show()

    def save_kernel(self, name):

        self.view_speed_build(False)
        plt.savefig(f"{self.sim_conf.kernel_path}KernelSpeed_{name}_{self.sim_conf.kernel_mode}.png")

        self.view_angle_build(False)
        plt.savefig(f"{self.sim_conf.kernel_path}KernelAngle_{name}_{self.sim_conf.kernel_mode}.png")


def build_track_kernel(conf):
    img = prepare_track_img(conf) 
    img, img2 = shrink_img(img, conf.track_shrink_pixels)
    kernel = KernelGenerator(img2, conf)
    kernel.view_kernel_angles(False)
    kernel.calculate_kernel(100)

    name = f"Kernel_{conf.kernel_mode}_{conf.map_name}"
    np.save(f"{conf.kernel_path}{name}.npy", kernel.kernel)
    print(f"Saved kernel to file: {name}")


def generate_kernels():
    conf = load_conf("config_file")
    build_dynamics_table(conf)

    conf.map_name = "columbia_small"
    build_track_kernel(conf)


def view_kernel():
    conf = load_conf("config_file")
    img = prepare_track_img(conf) 
    img, img2 = shrink_img(img, 5)
    k = VeiwKernel(conf, img2)

if __name__ == "__main__":

    # conf = load_conf("kernel_config")
    # build_track_kernel(conf)

    generate_kernels()

    # view_kernel()


