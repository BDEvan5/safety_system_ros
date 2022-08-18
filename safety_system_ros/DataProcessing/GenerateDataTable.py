import csv
from matplotlib import pyplot as plt 
import numpy as np
import yaml, glob, os
from PIL import Image
import tikzplotlib
import trajectory_planning_helpers as tph

class BagDataPlotter:
    def __init__(self, map_name):
        self.pos_xs = None
        self.pos_ys = None
        self.thetas = None
        self.vs = None
        self.N = None
        self.time_steps = None

        self.steering_ds = None

        self.path = None
        self.name = None

        self.near_start = True
        self.toggle_list = 0
        self.start_x = 0
        self.start_y = 0
        self.start_time = None

        self.lap_times = []

        self.map_name = map_name
        self.resolution = None
        self.origin = None
        self.map_img_name = None

        self.read_yaml_file()
        self.load_map()
        
    def read_yaml_file(self):
        file_name = 'map_data/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

        yaml_file = dict(documents.items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        self.stheta = -np.pi/2
        self.map_img_name = yaml_file['image']

    def load_map(self):
        map_img_name = 'map_data/' + self.map_img_name
        try:
            self.map_img = np.array(Image.open(map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        except Exception as e:
            print(f"MapPath: {map_img_name}")
            print(f"Exception in reading: {e}")
            raise ImportError(f"Cannot read map")
        try:
            self.map_img = self.map_img[:, :, 0]
        except:
            pass

    def load_csv_data(self, folder):
        track = []

        self.path = folder
        self.name = os.path.split(self.path)[-1]
        try:
            filename = glob.glob(self.path + "/*_odom.csv")[0]
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
                for lines in csvFile:  
                    track.append(lines)

            track = np.array(track)
            print(f"Track Loaded: {filename}")
        except Exception as e:
            print(f"Exception in reading: {e}")
            return 
            
        self.pos_xs = track[:,0]
        self.pos_ys = track[:,1]
        self.thetas = track[:,1]
        self.vs = track[:,3]

        self.time_steps = track[:,4]

        self.N = len(track)
            
        actions = []
        try:
            filename = glob.glob(self.path + "/*_actions.csv")[0]
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
                for lines in csvFile:  
                    actions.append(lines)

            actions = np.array(actions)
            self.steering_ds = np.array(actions[:, 1])
            print(f"Track Loaded: {filename}")
        except Exception as e:
            self.steering_ds = None
            print(f"Exception in reading: {e}")
            return 

        # self.show_poses()

        lap_steps = self.separate_laps()
        self.generate_table_data(lap_steps)
        # self.save_lap_times()
        # self.plot_lap_imgs(lap_steps)
        # self.plot_paper_laps_lobby(lap_steps)
        self.plot_paper_laps_levine(lap_steps)

    def save_lap_times(self):
        with open(self.path + "/lap_times.csv", 'w') as file:
            csvFile = csv.writer(file)
            # csvFile.writerow(["Lap", "Time"])
            for i in range(len(self.lap_times)):
                csvFile.writerow([i, self.lap_times[i]])

        print(f"Successfully saved lap times: {self.lap_times}")
        self.lap_times = []

    def plot_lap_imgs(self, lap_steps):
        for i in range(len(lap_steps)-1):
            plt.figure(i)
            plt.clf()
            plt.title(f"{self.name}: Lap {i}")
            plt.imshow(self.map_img, cmap='gray', origin='lower')
            start = lap_steps[i]
            end = lap_steps[i+1]
            pts = np.array([self.pos_xs[start:end], self.pos_ys[start:end]]).T
            xs, ys = self.convert_positions(pts)
            plt.plot(xs, ys)
            plt.pause(0.00011)
            print(f"{self.path}/{self.name}_lap_{i}.png")
            plt.savefig(f"{self.path}/{self.name}_lap_{i}.png")
            # plt.savefig(f"{self.path}/lap_{i+1}.svg")
            plt.close(i)
        plt.show()
        plt.close()

    def generate_table_data(self, lap_steps):
        pts = np.concatenate((self.pos_xs[:, np.newaxis], self.pos_ys[:, np.newaxis]), axis=1)
        for i in range(len(lap_steps)-1):
            start = lap_steps[i]
            end = lap_steps[i+1] 
            lap_pts = pts[start:end]
            ss = np.linalg.norm(np.diff(lap_pts, axis=0), axis=1)
            distance = np.sum(ss, axis=0)

            ths, ks = tph.calc_head_curv_num.calc_head_curv_num(pts[start:end], ss, False)
            filter_val = 0.5
            ks = np.clip(ks, -filter_val, filter_val)

            total_curvature = np.sum(np.abs(ks))

            print(f"Distance: {distance}")
            print(f"Total Curvature: {total_curvature}")

            if self.steering_ds is not None:
                start_time_x10 = int(self.lap_times[i]*10)
                end_time_x10 = int(min(self.lap_times[i+1]*10, len(self.steering_ds)-1))
                if end_time_x10 < 5+ start_time_x10:
                    continue
                mean_steer = np.abs(self.steering_ds[start_time_x10:end_time_x10]).mean()
                print(f"Mean Steer: {mean_steer}")
            print("---------------")


    def plot_paper_laps_levine(self, lap_steps):
        for i in range(len(lap_steps)-1):
            plt.figure(1)
            plt.clf()
            # plt.title(f"{self.name}: Lap {i}")
            plt.imshow(self.map_img, cmap='gray', origin='lower')
            # plt.xlim(40, 640)
            # plt.ylim(230, 540)
            start = lap_steps[i]
            end = lap_steps[i+1]
            pts = np.array([self.pos_xs[start:end], self.pos_ys[start:end]]).T
            xs, ys = self.convert_positions(pts)
            # plt.gcf()
            plt.plot(xs, ys, linewidth=2, color='darkblue')
            # for i, pt in enumerate(pts):
            #     if i%20 == 0:
            #         plt.plot(xs[0], ys[1], 'o', color='darkgreen')

            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect('equal', adjustable='box')

            plt.pause(0.00011)
            # print(f"{self.path}/{self.name}_lap_{i}.png")
            # path = f"/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/LobbyPaperLaps/{self.name}_lap_{i}"
            path = f"/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/LevinePaperLaps/{self.name}_lap_{i}"
            plt.savefig(path + ".png")

            tikzplotlib.save(path + ".tex", strict=True, extra_axis_parameters=['axis equal image', 'width=0.56\\textwidth'])
            plt.close(i)
        # plt.show()
        # plt.close()


    def plot_paper_laps_lobby(self, lap_steps):
        for i in range(len(lap_steps)-1):
            plt.figure(1)
            plt.clf()
            # plt.title(f"{self.name}: Lap {i}")
            plt.imshow(self.map_img.T, cmap='gray', origin='lower')
            # plt.xlim(40, 640)
            # plt.ylim(230, 540)
            start = lap_steps[i]
            end = lap_steps[i+1]
            pts = np.array([self.pos_xs[start:end], self.pos_ys[start:end]]).T
            xs, ys = self.convert_positions(pts)
            # plt.gcf()
            plt.plot(ys, xs, linewidth=2, color='darkblue')
            # plt.plot(xs, ys, linewidth=2, color='darkblue')

            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect('equal', adjustable='box')

            plt.pause(0.00011)
            path = f"/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/LobbyPaperLaps/{self.name}_lap_{i}"
            plt.savefig(path + ".png")

            tikzplotlib.save(path + ".tex", strict=True, extra_axis_parameters=['axis equal image', 'width=0.56\\textwidth'])
            plt.close(i)
        # plt.show()
        # plt.close()


    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        if c >= self.map_img.shape[1]:
            c = self.map_img.shape[1] - 1
        if r >= self.map_img.shape[0]:
            r = self.map_img.shape[0] - 1

        return c, r

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def separate_laps(self):
        lap_steps = []
        self.lap_times = []
        self.start_x = self.pos_xs[0]
        self.start_y = self.pos_ys[0]
        self.start_time = self.time_steps[0]
        for i, (x, y, th) in enumerate(zip(self.pos_xs, self.pos_ys, self.thetas)):
            pose = np.array([x, y, th])
            done = self.check_lap_done(pose)
            if self.vs[i] == 0.0:
                self.start_time = self.time_steps[i]
            if done or i == 0:
                lap_steps.append(i)
                self.near_start = True
                self.toggle_list = 0
                if (self.time_steps[i] - self.start_time) > 1:
                    self.lap_times.append(self.time_steps[i] - self.start_time)
                else:
                    self.lap_times.append(0)
                print(f"Done on step: {i} --> Laptime: {self.time_steps[i] - self.start_time}")
                self.start_time = self.time_steps[i]
            
        # if len(self.pos_xs) > i + 100: lap_steps.append(i)
        return lap_steps

    def check_lap_done(self, position):
        # start_theta = -1.6
        start_theta = 0
        start_rot = np.array([[np.cos(-start_theta), -np.sin(-start_theta)], [np.sin(-start_theta), np.cos(-start_theta)]])

        poses_x = np.array(position[0]) - self.start_x
        poses_y = np.array(position[1]) - self.start_y
        delta_pt = np.dot(start_rot, np.stack((poses_x, poses_y), axis=0))

        dist2 = delta_pt[0]**2 + delta_pt[1]**2
        closes = dist2 <= 2
        if closes and not self.near_start:
            self.near_start = True
            self.toggle_list += 1
        elif not closes and self.near_start:
            self.near_start = False
            self.toggle_list += 1
        done = self.toggle_list >= 2

        return done

    def show_poses(self):
        plt.figure(3)
        plt.clf()
        plt.title(f"Positions")
        plt.imshow(self.map_img, cmap='gray', origin='lower')
        pts = np.array([self.pos_xs, self.pos_ys]).T
        xs, ys = self.convert_positions(pts)
        plt.plot(xs, ys)

        plt.pause(1)
        # plt.pause(0.0001)
        # plt.show()

        plt.savefig(f"{self.path}/positions.png")
        plt.savefig(f"{self.path}/positions.svg")

from safety_system_ros.utils.util_functions import *
def explore_levine():
    path = "/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/LevinePaperLaps"
    init_file_struct(path)

    log = BagDataPlotter("levine_2nd")

    path = "Data/PaperData/UsefulLevine/" 

    folders = glob.glob(f"{path}*")
    for i, folder in enumerate(folders):
        print(f"Folder being opened: {folder}")
        log.load_csv_data(folder)
        print(f"Folder {i} done")


def explore_lobby():
    path = "/home/benjy/sim_ws/src/safety_system_ros/Data/PaperData/LobbyPaperLaps"
    init_file_struct(path)
    log = BagDataPlotter("lobby")

    path = "Data/PaperData/UsefulLobby/" 

    folders = glob.glob(f"{path}*")
    for i, folder in enumerate(folders):
        print(f"Folder being opened: {folder}")
        log.load_csv_data(folder)
        print(f"Folder {i} done")



if __name__ == '__main__':

    explore_lobby()
    # explore_levine()

    # plt.show()


