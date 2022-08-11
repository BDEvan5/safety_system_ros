import numpy as np
import tikzplotlib  
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def make_steering_graph(vehicle="PaperSafetyAgent_3"):
    directory = f'/home/benjy/sim_ws/src/safety_system_ros/Data/Vehicles/{vehicle}/'
    init_steering = np.load(directory + "/init_steering_actions.npy")
    safe_steering = np.load(directory + "/safe_steering_actions.npy")

    xs = np.arange(len(init_steering)) /100

    plt.figure(figsize=(8,2))
    # plt.plot(init_steering, 'b', label="Init Steering")
    plt.plot(xs, safe_steering, 'blue', label="Safe Steering")
    for i in range(len(init_steering)):
        if init_steering[i] != safe_steering[i]:
            plt.plot(i/100, init_steering[i], 'x', color='green')

    plt.plot([3.40, 3.40], [-0.4, 0.4], '--', color='grey')
    plt.plot([6.70, 6.70], [-0.4, 0.4], '--', color='grey')

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(0.25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Steering Angle")
    plt.xlim(0, 6.8)


    # plt.tight_layout()
    plt.grid()

    # plt.savefig(directory + "steering_graph.pdf")
    tikzplotlib.save(directory + "steering_graph.tex", strict=True, extra_axis_parameters=['width=0.9\\textwidth', 'height=4cm'])

    plt.show()

def plot_test_actions(vehicle="PaperSafetyAgent_2"):
    directory = f'/home/benjy/sim_ws/src/safety_system_ros/Data/Vehicles/{vehicle}/'
    steering = np.load(directory + "steering_actions.npy")

    plt.figure(figsize=(10,2))
    plt.plot(steering, 'b', label="Steering")

    plt.show()

def plot_training_rewards(vehicle="PaperSafetyAgent_3"):
    directory = f'/home/benjy/sim_ws/src/safety_system_ros/Data/Vehicles/{vehicle}/'
    rewards = np.load(directory + "ep_rewards.npy")

    plt.figure(figsize=(10,2))
    plt.plot(rewards, 'b', label="Rewards")

    plt.show()

make_steering_graph()
# plot_test_actions()
# plot_training_rewards()