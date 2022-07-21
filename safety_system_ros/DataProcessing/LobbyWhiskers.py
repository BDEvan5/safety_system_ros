import glob, csv 
import numpy as np 
import matplotlib.pyplot as plt
import os
import tikzplotlib

def load_data():
    path = "/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/UsefulLobby/"

    names = []
    time_sets = []

    folders = glob.glob(f"{path}*/")
    for i, folder in enumerate(folders):
        print(f"Folder being opened: {folder}")
        track = []

        name = os.path.split(folder)[-1]
        try:
            # filename = glob.glob(folder + "/lap_times.csv")
            filename = folder + "/lap_times.csv"
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
                for lines in csvFile:  
                    track.append(lines)

            track = np.array(track)
            print(f"Track Loaded: {filename}")
        except Exception as e:
            print(f"Exception in reading: {e}")
            raise ImportError

        time_sets.append(track[:, 1])
        names.append(name)

    return time_sets, names

def make_boxes():
    time_sets, names =  load_data()
    names = ['SSS', 'Baseline', 'PP']

    plt.figure(figsize=(4, 1.5))

    # for i in range(len(time_sets)):
    plt.boxplot(time_sets, labels=names, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'}, widths=0.7, showfliers=False)
    plt.grid(True)
    plt.xlim(9, 13.5)
    plt.tight_layout()
    plt.xlabel('Lap-time (seconds)')

    path = "/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/UsefulLobby/"
    tikzplotlib.save(path + "lobby_boxplot.tex", strict=True, extra_axis_parameters=['axis equal image', 'width=0.46\\textwidth'])

    plt.show()


make_boxes()


