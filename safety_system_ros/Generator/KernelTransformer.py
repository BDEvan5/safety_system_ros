import numpy as np
from safety_system_ros.utils.util_functions import *
import matplotlib.pyplot as plt

class Transformer:
    def __init__(self):
        self.conf = load_conf_mac("config_file")
        # self.conf.map_name = "f1_aut_wide"
        self.conf.map_name = "levine_2nd"


        self.dynamics = np.load(f"{self.conf.dynamics_path}_dyns.npy")
        print(f"Dynamics Loaded: {self.dynamics.shape}")

        kernel_std = np.load(f"{self.conf.kernel_path}Kernel_std_{self.conf.map_name}.npy")
        self.kernel = np.load(f"{self.conf.kernel_path}Kernel_filter_{self.conf.map_name}.npy")
        self.new_kernel = kernel_std

    def transform(self):
        self.new_kernel = transform(self.kernel, self.new_kernel, self.dynamics)

        np.save(f"{self.conf.kernel_path}Kernel_transform_{self.conf.map_name}.npy", self.new_kernel)

        print(f"Saved kernel to file: Kernel_transform_{self.conf.map_name}")

    def plot_kenrel_action(self):
        action = 0 # go straight from
        theta = 20
        # plot all the places where you can go straight from

        plt.figure(1)
        self.kernel = self.kernel.astype(np.int8)
        img = self.new_kernel[:, :, theta, action] + self.kernel[:, :, theta, 0]
        # img =  self.kernel[:, :, theta, 0]
        plt.imshow(img.T, origin='lower')
        plt.imshow(self.new_kernel[:, :, theta, action].T + self.kernel[:, :, theta, 0].T, origin='lower')

        plt.show()

@njit(cache=True)
def transform(kernel, new_kernel, dynamics):
    nx, ny, nth, nm = new_kernel.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nth):
                for l in range(nm):
                    # for q in range(nm):
                    di, dj, new_k, new_q = dynamics[k, 4, l, 0, :]

                    new_i = min(max(0, i + di), nx-1)  
                    new_j = min(max(0, j + dj), ny-1)

                    new_kernel[i, j, k, l] = kernel[new_i, new_j, new_k, 0]
    
    return new_kernel


if __name__ == "__main__":
    t = Transformer()
    t.transform()
    t.plot_kenrel_action()
