
import numpy as np
from safety_system_ros.utils.util_functions import *


class LapLogger:
    def __init__(self, vehicle_path):
        abs_path = "/home/benjy/sim_ws/src/safety_system_ros/"
        # self.path = abs_path + "Data/Logs/"
        self.path = vehicle_path + "/" # activate this after teting
        self.current_log_file = None

        self.lap = 0

        self.info_log_file = self.path + f"InfoLogging.txt"
    
        with open(self.info_log_file, "w") as f:
            print(f"{self.info_log_file} created")

        self.open_log_file()

    def open_log_file(self):
        self.current_log_file = f"{self.lap}_log_file.csv"
        with open(self.path + self.current_log_file, "w") as f:
            print(f"{self.current_log_file} created")
            # f.write("act_steer, act_vel, pos_x, pos_y, theta, v\n")

    def reset_logging(self):
        self.lap += 1 
        self.open_log_file()

    def write_env_log(self, data):
        with open(self.path + self.current_log_file, "a") as f:
            f.write(data + "\n")

    def write_info_log(self, log_text):
        with open(self.info_log_file, "a") as f:
            f.write(log_text + "\n")

    