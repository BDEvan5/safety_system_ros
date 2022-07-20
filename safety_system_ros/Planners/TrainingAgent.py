import numpy as np 
from safety_system_ros.utils.TD3 import TD3
from safety_system_ros.utils.HistoryStructs import TrainHistory
from safety_system_ros.utils.RewardFunctions import *
import torch
from numba import njit

from safety_system_ros.utils.util_functions import *

class BaseVehicle: 
    def __init__(self, agent_name, sim_conf):
        self.name = agent_name
        self.n_beams = 20
        self.max_v = sim_conf.max_v
        self.speed = sim_conf.vehicle_speed
        self.max_steer = sim_conf.max_steer
        self.range_finder_scale = sim_conf.range_finder_scale

        self.loop_counter = 0
        self.action = None
        self.v_min_plan =  sim_conf.v_min_plan


    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
        v_current = obs['state'][3]
        d_current = obs['state'][4]
        scan = np.array(obs["scan"]) 
        # print(scan)
        scan = scan[190:-190] # reduce to 700 beams
        inds = np.arange(0, 700, 35) # slice 20 beams
        scan = scan[inds]

        scan = np.clip(scan/self.range_finder_scale, 0, 1)

        cur_v = [v_current/self.max_v]
        cur_d = [d_current/self.max_steer]

        nn_obs = scan
        # nn_obs = np.concatenate([cur_v, cur_d, scan])

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer

        action = np.array([steering_angle, self.speed])

        return action

class TrainVehicle(BaseVehicle):
    def __init__(self, sim_conf, agent_name, load=False):
        super().__init__(agent_name, sim_conf)

        self.path = sim_conf.directory + sim_conf.vehicle_path + agent_name 
        init_file_struct(self.path)

        state_space = self.n_beams # + 2
        self.agent = TD3(state_space, 1, 1, agent_name)
        self.agent.try_load(load, sim_conf.h_size, self.path)

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.t_his = TrainHistory(agent_name, sim_conf, load)

    def plan(self, obs, add_mem_entry=True):
        nn_obs = self.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs, nn_obs)
            
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        self.state = obs
        nn_action = self.agent.act(nn_obs)
        self.nn_act = nn_action

        self.nn_state = nn_obs

        self.action = self.transform_action(nn_action)

        return self.action # implemented for the safety wrapper

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            # reward = self.calculate_reward(self.state, s_prime)
            reward = s_prime['reward']
    
            self.t_his.add_step_data(reward)

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = s_prime['reward']
        # reward = self.calculate_reward(self.state, s_prime)

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)
        # self.t_his.print_update(False) #remove this line
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(False)
        self.agent.save(self.path)
        self.state = None

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)

    def intervention_entry(self, s_prime):
        """
        To be called when the supervisor intervenes
        """
        nn_s_prime = self.transform_obs(s_prime)
        # reward = self.calculate_reward(self.state, s_prime)
        reward = s_prime['reward'] - 1


        self.t_his.add_step_data(reward)

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)

    def lap_complete(self):
        """
        To be called when ep is done.
        """
        print(f"Lap Complete: in agent {self.t_his.t_counter}")
        self.t_his.lap_done(False)
        self.t_his.print_update(False) #remove this line
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(False)
            self.agent.save(self.path)


class TestVehicle(BaseVehicle):
    def __init__(self, sim_conf, agent_name):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """

        super().__init__(agent_name, sim_conf)

        self.path = sim_conf.directory  + sim_conf.vehicle_path + agent_name
        self.actor = torch.load(self.path + '/' + agent_name + "_actor.pth")

        print(f"Agent loaded: {agent_name}")

    def plan(self, obs):
        nn_obs = self.transform_obs(obs)

        # if obs['state'][3] < self.v_min_plan:
        #     self.action = np.array([0, 7])
        #     return self.action

        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_action = self.actor(nn_obs).data.numpy().flatten()
        self.nn_act = nn_action

        self.action = self.transform_action(nn_action)

        return self.action # implemented for the safety wrapper

