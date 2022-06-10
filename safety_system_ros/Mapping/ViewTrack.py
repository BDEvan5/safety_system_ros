
import yaml 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import csv
import casadi as ca 
from scipy import ndimage 
import io

# import toy_auto_race.Utils.LibFunctions as lib
import LibFunctions as lib
import matplotlib.collections as mcoll
import matplotlib.path as mpath

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

class PreMap:
    def __init__(self, conf, map_name) -> None:
        self.conf = conf
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

        self.read_yaml_file()
        self.load_map()
        self.load_map_opti()

    def load_map(self):
        map_img_name = 'maps/' + self.map_img_name

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

    

    def load_map_opti(self):
        track = []
        filename = 'maps/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        # these get expanded
        self.wpts = track[:, 1:3]
        self.vs = track[:, 5]
        self.ss = track[:, 0]
        self.N = len(track)


    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)
      
    def read_yaml_file(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

        yaml_file = dict(documents.items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        # self.stheta = yaml_file['start_pose'][2]
        self.stheta = -np.pi/2
        self.map_img_name = yaml_file['image']

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        if c >= self.map_img.shape[1]:
            c = self.map_img.shape[1] - 1
        if r >= self.map_img.shape[0]:
            r = self.map_img.shape[0] - 1

        return c, r

    def make_track_picture(self):
        plt.figure(1)
        self.map_img[self.map_img == 0] = 95
        self.map_img[self.map_img == 255] = 50
        self.map_img[0, 0] = 100
        self.map_img[0, 1] = 0
        plt.imshow(self.map_img, cmap='gray', origin='lower')


        wpt_x, wpt_y = self.convert_positions(self.wpts)
        plt.plot(wpt_x, wpt_y, linewidth=5, color='darkblue')
        # colorline(wpt_x, wpt_y, self.vs, cmap='jet', linewidth=5)

        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        plt.savefig(f"Imgs/{self.map_name}_track.svg", bbox_inches='tight', pad_inches=0.1)

        plt.show()


if __name__ == "__main__":
    # map_name = 'f1_aut_wide'
    map_name = 'example_map'
    # map_name = 'columbia_small'

    fname = "config_test"
    conf = lib.load_conf(fname)

    pre_map = PreMap(conf, map_name)
    pre_map.make_track_picture()