import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from safety_system_ros.utils.util_functions import load_conf_mac
from KernelGenerator import prepare_track_img, shrink_img

class VeiwKernel:
    def __init__(self, conf, track_img):
        if conf.steering:
            kernel_name = f"{conf.kernel_path}Kernel_std_{conf.map_name}.npy"
        else:
            kernel_name = f"{conf.kernel_path}Kernel_filter_{conf.map_name}.npy"
        self.kernel = np.load(kernel_name)

        self.o_map = np.copy(track_img)    
        self.fig, self.axs = plt.subplots(2, 2)

        
        self.phis = np.linspace(-conf.phi_range/2, conf.phi_range/2, conf.n_phi)

        self.qs = np.linspace(-conf.max_steer, conf.max_steer, conf.nq_steer)
        # self.view_speed_build(True)
        self.view_kernel_angles(True)
     
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

    def view_kernel_angles(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        mode_ind = 0

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


def view_kernel():
    conf = load_conf_mac("config_file")
    conf.map_name = "levine_2nd"
    # conf.map_name = "f1_aut_wide"
    img = prepare_track_img(conf) 
    img, img2 = shrink_img(img, 5)
    k = VeiwKernel(conf, img2)

if __name__ == "__main__":


    view_kernel()


