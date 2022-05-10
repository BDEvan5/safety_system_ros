
from SuperSafety.Utils.utils import limit_phi, load_conf

from SuperSafety.Supervisor.Dynamics import run_dynamics_update

import numpy as np
from matplotlib import pyplot as plt

import numpy as np
from numba import njit


class Modes:
    def __init__(self, conf) -> None:
        self.time_step = conf.kernel_time_step
        self.nq_steer = conf.nq_steer
        self.max_steer = conf.max_steer
        vehicle_speed = conf.vehicle_speed
        wheelbase = conf.l_r + conf.l_f

        self.ds = np.linspace(-self.max_steer, self.max_steer, self.nq_steer)
        phi_dots = vehicle_speed/wheelbase * np.tan(self.ds)
        vs = vehicle_speed * np.ones_like(phi_dots)
        self.qs = np.stack((phi_dots, vs), axis=1)
        self.acts = np.stack((self.ds, vs), axis=1)

        self.n_modes = len(self.qs)

    def get_mode_id(self, ang_z):
        d_ind = np.argmin(np.abs(self.qs[:, 0] - ang_z))
        
        return int(d_ind)

    def action2mode(self, action):
        id = self.get_mode_id(action[0])
        return self.qs[id]

    def __len__(self): return self.n_modes


def generate_dynamics_entry(state, action, m, time, resolution, phis):
    dyns = np.zeros(4)
    new_state = run_dynamics_update(state, action, time)
    dx, dy, phi, vel, ang_z = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
    new_q = m.get_mode_id(ang_z)

    phi = limit_phi(phi)
    new_k = int(round((phi + np.pi) / (2*np.pi) * (len(phis)-1)))
    dyns[2] = min(max(0, new_k), len(phis)-1)
    dyns[0] = int(round(dx * resolution))                  
    dyns[1] = int(round(dy * resolution))                  
    dyns[3] = int(new_q)       

    return dyns           




# @njit(cache=True)
def build_viability_dynamics(m, conf):
    phis = np.linspace(-np.pi, np.pi, conf.n_phi)

    ns = conf.n_intermediate_pts
    dt = conf.kernel_time_step / ns

    dynamics = np.zeros((len(phis), len(m), len(m), ns, 4), dtype=np.int)
    invalid_counter = 0
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(m.acts): # searches through actions

                for l in range(ns):
                    dynamics[i, j, k, l] = generate_dynamics_entry(state.copy(), action, m, dt*(l+1), conf.n_dx, phis)                             
                
    print(f"Invalid transitions: {invalid_counter}")
    print(f"Dynamics Table has been built: {dynamics.shape}")

    return dynamics




def build_dynamics_table(sim_conf):
    m = Modes(sim_conf)

    if sim_conf.kernel_mode == "viab":
        dynamics = build_viability_dynamics(m, sim_conf)
    else:
        raise ValueError(f"Unknown kernel mode: {sim_conf.kernel_mode}")


    np.save(f"{sim_conf.dynamics_path}{sim_conf.kernel_mode}_dyns.npy", dynamics)

from matplotlib.ticker import MultipleLocator

def view_dynamics_table(sim_conf):
    dynamics = np.load(f"{sim_conf.dynamics_path}{sim_conf.kernel_mode}_dyns.npy")

    print(f"Dynamics Loaded: {dynamics.shape}")

    for mode in range(9):
        dyns = dynamics[30, mode, :, 1, :]
        print(f"Dynamics Loaded: {dyns.shape}")
        print(dyns)
        angles = np.linspace(-np.pi, np.pi, 41)
        a = [angles[i] for i in dyns[:, 2]]

        spacing = 2.5
        minorLocator = MultipleLocator(spacing)

        plt.figure(1, figsize=(1.9, 1.9))
        plt.clf()
        plt.ylim([-2, 28])
        plt.gca().xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, ls='--', which='both')
        plt.plot(dyns[:, 0], dyns[:, 1], '.', color='darkblue', markersize=10)
        plt.plot(0, 0, '.', color='red', markersize=10)
        l = 2
        plt.arrow(0, 0, 0, l, head_width=0.24, head_length=0.5, fc='red', ec='red', width=0.06)
        dx = np.cos(a) * l
        dy = np.sin(a) * l
        for i in range(9):
            plt.arrow(dyns[i, 0], dyns[i, 1], dx[i], dy[i], head_width=0.24, head_length=0.5, width=0.06, fc='darkblue', ec='darkblue')

        plt.tight_layout()
        plt.savefig("Data/Modes/mode_" + str(mode) + ".png", bbox_inches='tight', pad_inches=0.0)


if __name__ == "__main__":
    conf = load_conf("config_file")

    build_dynamics_table(conf)

    view_dynamics_table(conf)

