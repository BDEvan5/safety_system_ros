import csv
from matplotlib import pyplot as plt 
import numpy as np
import yaml
from PIL import Image

class Logggs:
    def __init__(self, path):
        self.steers = None
        self.vels = None
        self.pos_xs = None
        self.pos_ys = None
        self.thetas = None
        self.vs = None
        self.N = None

        self.path = path

        self.resolution = None
        # self.map_name = "f1_aut_wide"
        self.map_name = "columbia_small"
        # self.map_name = "example_map"
        self.origin = None
        self.map_img_name = None

        self.t = Trajectory(self.map_name)

        self.height = None
        self.width = None

        self.read_yaml_file()
        self.load_map()

        
    def read_yaml_file(self):
        file_name = 'map_data/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

        yaml_file = dict(documents.items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        # self.stheta = yaml_file['start_pose'][2]
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

        self.height = self.map_img.shape[1]
        self.width = self.map_img.shape[0]



    def load_csv_log(self, lap_n):
        track = []
        filename = self.path + f"{lap_n}_log_file.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        # these get expanded
        self.steers = track[:,0]
        self.vels = track[:,1]
        self.pos_xs = track[:,2]
        self.pos_ys = track[:,3]
        self.thetas = track[:,4]
        self.vs = track[:,5]

        self.N = len(track)

        self.m = 3.47
        self.L = 0.33

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


    def show_poses(self):
        plt.figure(3)
        plt.title(f"Positions")
        plt.imshow(self.map_img, cmap='gray', origin='lower')
        # plt.plot(self.pos_xs, self.pos_ys)
        pts = np.array([self.pos_xs, self.pos_ys]).T
        xs, ys = self.convert_positions(pts)
        plt.plot(xs, ys)

        xs, ys = self.convert_positions(self.t.waypoints)
        plt.plot(xs, ys, 'r')

        plt.pause(0.0001)
        # plt.show()

    def plot_vels(self):
        plt.figure()
        plt.title(f"Vels and steering")
        plt.plot(self.vels)
        plt.plot(self.vs)
        # steering = np.abs(self.steers) * 5
        # plt.plot(steering)
        plt.pause(0.0001)
        # plt.show()

    def plot_forces(self):
        plt.figure()
        plt.title(f"Forces")
        f_lat = np.abs(self.vels ** 2  * np.tan(self.steers) * self.m/ self.L)
        f_long = np.abs(self.m * np.diff(self.vels) * 20) 

        f_total = (f_lat[:-1]**2 + f_long ** 2)**0.5


        plt.plot(f_lat)
        # plt.plot(f_long)
        # plt.plot(f_total, linewidth=2)
        # plt.legend(["f_lat", "f_long", "f_total"])

        plt.ylim(0, 40)

        plt.pause(0.0001)
        # plt.show()

class Trajectory:
    def __init__(self, map_name):
        self.map_name = map_name
        self.waypoints = None
        self.vs = None
        self.load_csv_track()
        self.n_wpts = len(self.waypoints)

        self.max_reacquire = 20

        self.diffs = None 
        self.l2s = None 
        self.ss = None 
        self.o_points = None

    def load_csv_track(self):
        track = []
        filename = 'map_data/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        # these get expanded
        self.waypoints = track[:, 1:3]
        self.vs = track[:, 5]



if __name__ == '__main__':
    log = Logggs('Data/Vehicles/TestingAgent_1/')
    log.load_csv_log(1)
    log.show_poses()
    # log.plot_vels()
    # log.plot_forces()


    plt.show()
