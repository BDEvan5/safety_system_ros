import numpy as np


class RandomPlanner:
    def __init__(self, conf, name="RandoPlanner"):
        self.d_max = conf.max_steer # radians  
        self.name = name
        self.speed = conf.vehicle_speed

    def plan(self, pos):
        #! TODO: change this to a normal distribution
        steering = np.random.normal(0, 0.1)
        steering = np.clip(steering, -self.d_max, self.d_max)
        # steering = np.random.uniform(-self.d_max, self.d_max)
        return np.array([steering, self.speed])

