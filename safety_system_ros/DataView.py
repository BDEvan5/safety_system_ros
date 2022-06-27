import csv 
import numpy as np 
from matplotlib import pyplot as plt


def load_and_view():

    data = []
    filename = f"Data/Vehicles/TrainingData/step_10.csv"
    # filename = f"Data/Vehicles/TrainingData/training_steps_9.csv"
    with open(filename, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
    
        for lines in csvFile:  
            data.append(lines)

        data = np.array(data)
        print(f"Track Loaded: {filename}")

    return data 

def view_rewards(data):
    n = len(data)

    cumsum = np.cumsum(data[:, 1])
    plt.figure(1)
    plt.plot(cumsum)

    plt.show()


data = load_and_view()
view_rewards(data)