import numpy as np
import tikzplotlib  
import matplotlib.pyplot as plt

def make_steering_graph(vehicle="PaperSafetyAgent_3"):
    directory = f'/home/benjy/sim_ws/src/safety_system_ros/Data/Vehicles/{vehicle}/'
    init_steering = np.load(directory + "/init_steering_actions.npy")
    safe_steering = np.load(directory + "/safe_steering_actions.npy")

    plt.figure(figsize=(8,2))
    # plt.plot(init_steering, 'b', label="Init Steering")
    plt.plot(safe_steering, 'r', label="Safe Steering")
    for i in range(len(init_steering)):
        if init_steering[i] != safe_steering[i]:
            plt.plot(i, init_steering[i], 'bx')

    plt.plot([340, 340], [-0.4, 0.4], '--', color='grey')
    plt.plot([670, 670], [-0.4, 0.4], '--', color='grey')

    plt.xlabel("Planning Steps")
    plt.ylabel("Steering Angle")
    plt.xlim(0, 680)

    tikzplotlib.save(directory + "steering_graph.tex", strict=True, extra_axis_parameters=['width=0.9\\textwidth'])

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

# make_steering_graph()
# plot_test_actions()
plot_training_rewards()