import os, shutil
import csv
import numpy as np
from matplotlib import pyplot as plt

SIZE = 20000


def plot_data(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = moving_average(values, moving_avg_period)
    plt.plot(moving_avg)    
    moving_avg = moving_average(values, moving_avg_period * 5)
    plt.plot(moving_avg)    
    plt.pause(0.001)

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'same') / period

class TrainHistory():
    def __init__(self, agent_name, conf, load=False) -> None:
        self.agent_name = agent_name
        self.path = conf.vehicle_path + self.agent_name 

        # training data
        self.ptr = 0
        self.lengths = np.zeros(SIZE)
        self.rewards = np.zeros(SIZE) 
        self.t_counter = 0 # total steps
        self.step_rewards = []
        
        # espisode data
        self.ep_counter = 0 # ep steps
        self.ep_reward = 0
        self.ep_rewards = []

        if not load:
            self.init_file_struct()

    def init_file_struct(self):
        path = os.getcwd() +'/' + self.path
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)

    def add_step_data(self, new_r):
        self.ep_reward += new_r
        self.ep_rewards.append(new_r)
        self.ep_counter += 1
        self.t_counter += 1 
        self.step_rewards.append(new_r)

    def lap_done(self, show_reward=False):
        self.lengths[self.ptr] = self.ep_counter
        self.rewards[self.ptr] = self.ep_reward
        # print(f"EP reward: {self.ep_reward:.2f}")
        self.ptr += 1

        if show_reward:
            plt.figure(8)
            plt.clf()
            plt.plot(self.ep_rewards)
            plt.plot(self.ep_rewards, 'x', markersize=10)
            plt.title(f"Ep rewards: total: {self.ep_reward:.4f}")
            plt.ylim([-1.1, 1.5])
            plt.pause(0.0001)

        self.ep_counter = 0
        self.ep_reward = 0
        self.ep_rewards = []


    def print_update(self, plot_reward=True):
        if self.ptr < 10:
            return
        
        mean10 = np.mean(self.rewards[self.ptr-10:self.ptr])
        mean100 = np.mean(self.rewards[max(0, self.ptr-100):self.ptr])
        # score = moving_average(self.rewards[self.ptr-100:self.ptr], 10)
        print(f"Run: {self.t_counter} --> Moving10: {mean10:.2f} --> Moving100: {mean100:.2f}  ")
        
        if plot_reward:
            # raise NotImplementedError
            plot_data(self.rewards[0:self.ptr], figure_n=2)

    def save_csv_data(self):
        data = []
        for i in range(len(self.rewards)):
            data.append([i, self.rewards[i], self.lengths[i]])
        full_name = self.path + '/training_data.csv'
        with open(full_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        data = []
        for i in range(len(self.step_rewards)):
            data.append([i, self.step_rewards[i]])
        full_name = self.path + '/step_data.csv'
        with open(full_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        plot_data(self.rewards[0:self.ptr], figure_n=2)
        plt.figure(2)
        plt.savefig(self.path + "/training_rewards.png")

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'same') / period

class RewardAnalyser:
    def __init__(self) -> None:
        self.rewards = []
        self.t = 0

    def add_reward(self, new_r):
        self.rewards.append(new_r)
        self.t += 1

    def show_rewards(self, show=False):
        plt.figure(6)
        plt.plot(self.rewards, '-*')
        plt.ylim([-1, 1])
        plt.title('Reward History')
        if show:
            plt.show()
