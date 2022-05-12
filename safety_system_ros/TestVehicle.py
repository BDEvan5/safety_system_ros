from safety_system_ros.Supervisor import Supervisor
from safety_system_ros.BaseNode import BaseNode
from safety_system_ros.Planners.TrainingAgent import TestVehicle
from safety_system_ros.Planners.PurePursuitPlanner import PurePursuitPlanner

import rclpy
import numpy as np

from safety_system_ros.utils.util_functions import *


class TestingNode(BaseNode):
    def __init__(self):
        conf = load_conf("config_file")

        super().__init__('test_vehicle', conf)

        
        self.planner = TestVehicle("SafetyTrainingAgent_2", conf) 
        # self.planner = RandomPlanner(conf)
        # self.planner = PurePursuitPlanner(conf)

        self.supervision = True 
        # self.supervision = False
        self.supervisor = Supervisor(conf)

    def calculate_action(self, observation):
        action = self.planner.plan(observation)
        if self.supervision: 
            return self.supervisor.supervise(observation['state'], action)
        return action

    def lap_complete_callback(self):
        print(f"Interventions: {self.supervisor.interventions}")

class RandomPlanner:
    def __init__(self, conf, name="RandoPlanner"):
        self.d_max = conf.max_steer # radians  
        self.name = name
        self.speed = conf.vehicle_speed

    def plan(self, pos):
        steering = np.random.uniform(-self.d_max, self.d_max)
        return np.array([steering, self.speed])



def main(args=None):
    rclpy.init(args=args)
    node = TestingNode()
    node.run_lap()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


