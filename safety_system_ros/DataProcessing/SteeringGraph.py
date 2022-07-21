import numpy as np
import tikzplotlib  
import matplotlib.pyplot as plt

def make_steering_graph(vehicle="PaperSafetyAgent_2"):
    directory = f'/home/benjy/sim_ws/src/safety_system_ros/Data/Vehicles/{vehicle}/'
    init_steering = np.load(directory + "/init_steering_actions.npy")
    safe_steering = np.load(directory + "/safe_steering_actions.npy")

    plt.figure(figsize=(10,2))
    plt.plot(init_steering, 'b', label="Init Steering")
    plt.plot(safe_steering, 'r', label="Safe Steering")

    plt.show()

def plot_test_actions(vehicle="PaperSafetyAgent_2"):
    directory = f'/home/benjy/sim_ws/src/safety_system_ros/Data/Vehicles/{vehicle}/'
    steering = np.load(directory + "steering_actions.npy")

    plt.figure(figsize=(10,2))
    plt.plot(steering, 'b', label="Steering")

    plt.show()



make_steering_graph()
# plot_test_actions()