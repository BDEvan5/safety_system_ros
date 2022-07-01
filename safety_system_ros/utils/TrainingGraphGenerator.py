import numpy as np 
import matplotlib.pyplot as plt
import tikzplotlib
import csv

directory =  "/Users/benjamin/Documents/GitHub/safety_system_ros/Data/Vehicles/"
def load_csv_data(filename):
    track = []
    with open(directory+filename, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
    
        for lines in csvFile:  
            track.append(lines)

    track = np.array(track)
    print(f"Track Loaded: {filename}")

    data = track[:, 1]

    return data

def moving_average(data, period):
    ret = np.convolve(data, np.ones(period), 'same') / period
    for i in range(period):
        # t = np.mean
        t = np.convolve(data, np.ones(i+1), 'valid') / (i+1)
        ret[i] = t[0]
        ret[-i-1] = t[-1]
    return ret

def make_paper_plot_safety():
    # path = "experiment_4/step_data.csv"
    # path = "experiment_1/step_data.csv"

    # path = "LevineT_1/step_data.csv"
    path = "experiment_2/step_data.csv"
    data = load_csv_data(path)

    i = 0
    avg_reward = []
    while i < len(data):
        avg = np.sum(data[i:i+20])
        avg_reward.append(avg)
        i += 20

    plt.figure(1, figsize=(5, 2))
    plt.plot(avg_reward, color='darkblue', linewidth=2)
    plt.plot(moving_average(avg_reward, 10), color='red', linewidth=2)

    plt.xlabel("Training Steps (x20)")
    plt.ylabel("Reward (20 Steps)")

    plt.tight_layout()
    plt.grid()

    tikzplotlib.save(directory + "experiment_2/reward_plot.tex", strict=True, extra_axis_parameters=['axis equal image', 'width=0.56\textwidth'])

    plt.show()


def make_paper_plot_baseline():
    path = "LevineT_1/training_data.csv"
    data = load_csv_data(path)
    data = data[np.abs(data)>0]
    

    # i = 0
    # avg_reward = []
    # while i < len(data):
    #     avg = np.sum(data[i:i+20])
    #     avg_reward.append(avg)
    #     i += 20

    plt.figure(1, figsize=(5, 2))
    plt.plot(data, color='darkblue', linewidth=2)
    plt.plot(moving_average(data, 10), color='red', linewidth=2)

    plt.xlabel("Training Episodes")
    plt.ylabel("Episode Reward")

    plt.tight_layout()
    plt.grid()

    tikzplotlib.save(directory + "LevineT_1/reward_plot.tex", strict=True, extra_axis_parameters=['axis equal image', 'width=0.56\textwidth'])

    plt.show()


make_paper_plot_baseline()
# make_paper_plot_safety()

