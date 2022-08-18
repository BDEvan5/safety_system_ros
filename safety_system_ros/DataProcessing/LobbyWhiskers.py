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

    print(names)
    print(time_sets)
    # names = ['SSS', 'Baseline', 'PP']

    return time_sets, names

def load_sim_data():
    baseline = np.load("Data/BaselineLapData.npy")
    safety = np.load("Data/Safety2LapData.npy")
    # pp = np.load("Data/PPLapData.npy")

    time_sets = [baseline, safety]

    names = ['Baseline', 'SSS']

    plt.figure(figsize=(4, 1.5))

    # for i in range(len(time_sets)):
    plt.boxplot(time_sets, labels=names, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'}, widths=0.7, showfliers=False, patch_artist=True)
    plt.grid(True)
    plt.xlim(9, 13.5)
    plt.tight_layout()
    plt.xlabel('Lap-time (seconds)')

    path = "/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/"
    tikzplotlib.save(path + "lobby_sim_boxplot.tex", strict=True, extra_axis_parameters=['width=0.46\\textwidth'])

    plt.show()

def combined_blox_plot():
    baseline = np.load("Data/BaselineLapData.npy")
    safety = np.load("Data/Safety2LapData.npy")
    # pp = np.load("Data/PPLapData.npy")

    time_sets2 = [baseline, safety]
    names = ['Baseline', 'SSS']

    time_sets2, names =  load_data()
    names = ['SSS', 'Baseline', 'PP']

    time_sets = [baseline, time_sets2[1], safety, time_sets2[0]]
    names = ['Sim: Base', 'Real: Base', 'Sim: SSS', 'Real: SSS']

    fig = plt.figure(figsize=(4, 1))
    ax = fig.add_subplot(111)

    # for i in range(len(time_sets)):
    positions = [0.7, 1.25, 2.25, 2.8]
    bp = ax.boxplot(time_sets, labels=names, vert=False, positions=positions, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'}, widths=0.5, showfliers=False, patch_artist=True)

    colors = ['green', 'blue', 'green', 'blue']
    # colors = ['#FFFF00', '#FF00FF', '#FFFF00', '#FF00FF']
    # borders = ['#FFFF00', '#FF00FF', '#FFFF00', '#FF00FF']
    borders = ['darkgreen', 'darkblue', 'darkgreen', 'darkblue']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_edgecolor(borders[i])

    for i, line in enumerate(bp['whiskers']):
        # line.set_linewidth(3)
        line.set(color=borders[int(i/2)])
    for i, line in enumerate(bp['caps']):
        # line.set_linewidth(3)
        line.set(color=borders[int(i/2)])

    
    for i, line in enumerate(bp['medians']):
        # line.set_linewidth(3)
        line.set(color=borders[int(i)])

    plt.grid(True)
    plt.xlim(9, 13.5)
    plt.tight_layout()
    plt.xlabel('Lap-time (seconds)')

    path = "/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/"
    tikzplotlib.save(path + "lobby_combined_boxplot.tex", strict=True, extra_axis_parameters=['axis equal image', 'width=0.46\\textwidth', 'height=4cm'])

    plt.show()




def make_boxes():
    time_sets, names =  load_data()
    names = ['SSS', 'Baseline', 'PP']

    plt.figure(figsize=(4, 1.5))

    # for i in range(len(time_sets)):
    plt.boxplot(time_sets, labels=names, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'}, widths=0.7, showfliers=False, patch_artist=True)
    plt.grid(True)
    plt.xlim(9, 13.5)
    plt.tight_layout()
    plt.xlabel('Lap-time (seconds)')

    path = "/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/UsefulLobby/"
    tikzplotlib.save(path + "lobby_boxplot.tex", strict=True, extra_axis_parameters=['axis equal image', 'width=0.46\\textwidth'])

    plt.show()


# make_boxes()
# load_sim_data()
combined_blox_plot()

