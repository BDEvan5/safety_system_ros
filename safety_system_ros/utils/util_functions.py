import yaml 
from argparse import Namespace
import math 
import numpy as np
from numba import njit

import os, shutil

def load_conf(fname):
    # mac_path = "/Users/benjamin/Documents/GitHub/safety_system_ros/config/"

    full_path =  "/home/benjy/sim_ws/src/safety_system_ros/config/" + fname + '.yaml'
    # full_path =  "/home/nvidia/f1tenth_ws/src/safety_system_ros/config/" + fname + '.yaml'
    # full_path =  mac_path + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    # np.random.seed(conf.random_seed)

    return conf

def load_conf_mac(fname):
    mac_path = "/Users/benjamin/Documents/GitHub/safety_system_ros/config/"

    # full_path =  "/home/benjy/sim_ws/src/safety_system_ros/config/" + fname + '.yaml'
    # full_path =  "/home/nvidia/f1tenth_ws/src/safety_system_ros/config/" + fname + '.yaml'
    full_path =  mac_path + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    # np.random.seed(conf.random_seed)

    return conf


def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)


def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z


def orientation_to_angle(orientation):
    x, y, z = quaternion_to_euler_angle(orientation.w, orientation.x, orientation.y, orientation.z)
    theta = z * np.pi / 180

    return theta


@njit(cache=True)
def limit_phi(phi):
    while phi > np.pi:
        phi = phi - 2*np.pi
    while phi < -np.pi:
        phi = phi + 2*np.pi
    return phi


@njit(cache=True)
def calculate_speed(delta):
    b = 0.523
    g = 9.81
    l_d = 0.329
    f_s = 0.8
    max_v = 4

    if abs(delta) < 0.06:
        return max_v

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    return V



