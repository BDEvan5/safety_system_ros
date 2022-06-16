import yaml 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import csv
# import casadi as ca 
from scipy import ndimage 
import io

# import toy_auto_race.Utils.LibFunctions as lib
import LibFunctions as lib


class ProcessMap:
    def __init__(self, conf, map_name) -> None:
        self.conf = conf #TODO: update to use new config style
        self.map_name = map_name

        self.map_img = None
        self.origin = None
        self.resolution = None
        self.stheta = None
        self.map_img_name = None

        self.cline = None
        self.nvecs = None
        self.widths = None

        self.wpts = None
        self.vs = None

    def run_conversion(self):
        self.read_yaml_file()
        self.load_map()

        self.render_map(True)
         
        # plt.figure(2)
        # plt.imshow(self.map_img, origin='lower')
        # plt.pause(0.0001)

        # self.dt = ndimage.distance_transform_edt(self.map_img) 
        # self.dt = np.array(self.dt *self.resolution)
    
    def render_map(self, wait=False):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, origin='lower')

        x, y = self.xy_to_row_column([0, 0])
        plt.plot(x, y, 'x', markersize=15)

        # ns = self.nvecs 
        # ws = self.widths
        # l_line = self.cline - np.array([ns[:, 0] * ws[:, 0], ns[:, 1] * ws[:, 0]]).T
        # r_line = self.cline + np.array([ns[:, 0] * ws[:, 1], ns[:, 1] * ws[:, 1]]).T


        # cx, cy = self.convert_positions(self.cline)
        # plt.plot(cx, cy, '--', linewidth=2)
        # lx, ly = self.convert_positions(l_line)
        # plt.plot(lx, ly, linewidth=1)
        # rx, ry = self.convert_positions(r_line)
        # plt.plot(rx, ry, linewidth=1)

        # for i, pt in enumerate(self.cline):
        #     plt.plot([lx[i], rx[i]], [ly[i], ry[i]])

        # if self.wpts is not None:
        #     wpt_x, wpt_y = self.convert_positions(self.wpts)
        #     plt.plot(wpt_x, wpt_y, linewidth=2)

        # plt.show()
        plt.pause(0.0001)
        if wait:
            plt.show()

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        if c >= self.map_img.shape[1]:
            c = self.dt.shape[1] - 1
        if r >= self.map_img.shape[0]:
            r = self.dt.shape[0] - 1

        return c, r

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def load_track_pts(self):
        track = []
        filename = 'map_data/' + self.name + "_std.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename} in env_map")

        self.N = len(track)
        self.track_pts = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.ws = track[:, 4:6]
        
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



def run_pre_map():
    fname = "config_file"
    conf = lib.load_conf(fname)
    # map_name = "porto"
    # map_name = "berlin"
    # map_name = "race_track"
    
    # map_name = "f1_aut_wide"
    # map_name = "example_map"
    # map_name = "levine_blocked"
    # map_name = "columbia_small"

    map_name = "levine_2nd"

    pre_map = ProcessMap(conf, map_name)
    pre_map.run_conversion()
    # pre_map.run_opti()


if __name__ == "__main__":
    run_pre_map()

