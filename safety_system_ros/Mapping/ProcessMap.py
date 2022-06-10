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
         
        # plt.figure(2)
        # plt.imshow(self.map_img, origin='lower')
        # plt.pause(0.0001)

        self.dt = ndimage.distance_transform_edt(self.map_img) 
        self.dt = np.array(self.dt *self.resolution)

        self.find_centerline(False)
        self.find_nvecs_old()
        # self.find_nvecs()
        self.set_true_widths()
        self.remove_crossing()
        self.remove_crossing()
        self.remove_crossing()
        self.render_map()

        self.save_map_std()

        self.render_map(True)

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

    def remove_crossing(self):
        ns = self.nvecs 
        ws = self.widths #* 1.1
        l_line = self.cline - np.array([ns[:, 0] * ws[:, 0], ns[:, 1] * ws[:, 0]]).T
        r_line = self.cline + np.array([ns[:, 0] * ws[:, 1], ns[:, 1] * ws[:, 1]]).T

        cs, ns, ws = [], [], []

        # cs.append(self.cline[0])
        # ns.append(self.nvecs[0])
        # ws.append(self.widths[0])
        skps = 0
        for i in range(self.N-1):
            pt_1_l = l_line[i]
            pt_1_r = r_line[i]
            pt_2_l = l_line[i+1]
            pt_2_r = r_line[i+1]

            if not lines_cross(pt_1_l, pt_1_r, pt_2_l, pt_2_r):
                cs.append(self.cline[i])
                ns.append(self.nvecs[i])
                ws.append(self.widths[i])
            else:
                skps += 1
                print(f"Crossing at {i}")

            # if lines_cross(pt_1_l, pt_1_r, pt_2_l, pt_2_r):
            #     print(f"Removing Line: {i}")
            #     self.cline = np.delete(self.cline, i, 0)
            #     self.nvecs = np.delete(self.nvecs, i, 0)
            #     self.widths = np.delete(self.widths, i, 0)
            #     # i -= 1
            #     l_line = self.cline - np.array([self.nvecs [:, 0] * self.nvecs [:, 0], self.nvecs[:, 1] * self.widths[:, 0]]).T
            #     r_line = self.cline + np.array([self.nvecs [:, 0] * self.widths[:, 1], self.nvecs [:, 1] * self.widths[:, 1]]).T
            #     self.N = len(self.cline)

        cs.append(self.cline[-1])
        ns.append(self.nvecs[-1])
        ws.append(self.widths[-1])

        self.cline = np.array(cs)
        self.nvecs = np.array(ns)
        self.widths = np.array(ws)
        self.N = len(self.cline)

    def find_centerline(self, show=True):
        dt = self.dt

        # d_search = 1
        d_search = 0.8
        n_search = 21
        dth = (np.pi * 4/5) / (n_search-1)

        # makes a list of search locations
        search_list = []
        for i in range(n_search):
            th = -np.pi/2 + dth * i
            x = -np.sin(th) * d_search
            y = np.cos(th) * d_search
            loc = [x, y]
            search_list.append(loc)

        pt = start = np.array([0, 0]) #TODO: start from map position
        self.cline = [pt]
        # th = self.stheta
        # th = -np.pi
        th = 0
        while (lib.get_distance(pt, start) > d_search/2 or len(self.cline) < 10) and len(self.cline) < 500:
            vals = []
            self.search_space = []
            for i in range(n_search):
                d_loc = lib.transform_coords(search_list[i], -th)
                search_loc = lib.add_locations(pt, d_loc)

                self.search_space.append(search_loc)

                x, y = self.xy_to_row_column(search_loc)
                val = dt[y, x]
                vals.append(val)

            ind = np.argmax(vals)
            d_loc = lib.transform_coords(search_list[ind], -th)
            pt = lib.add_locations(pt, d_loc)
            self.cline.append(pt)

            if show:
                self.plot_raceline_finding()
                # plt.show()

            th = lib.get_bearing(self.cline[-2], pt)
            print(f"Adding pt: {pt}")

        self.cline = np.array(self.cline)
        self.N = len(self.cline)
        print(f"Raceline found --> n: {len(self.cline)}")
        if show:
            self.plot_raceline_finding(True)
        # self.plot_raceline_finding(False)

    def plot_raceline_finding(self, wait=False):
        plt.figure(1)
        plt.clf()
        plt.imshow(self.dt, origin='lower')

        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            plt.plot(s_x, s_y, '+', markersize=16)

        for pt in self.search_space:
            s_x, s_y = self.xy_to_row_column(pt)
            plt.plot(s_x, s_y, 'x', markersize=12)


        plt.pause(0.001)

        if wait:
            plt.show()

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        if c >= self.dt.shape[1]:
            c = self.dt.shape[1] - 1
        if r >= self.dt.shape[0]:
            r = self.dt.shape[0] - 1

        return c, r

    def find_nvecs(self):
        N = len(self.cline)

        n_search = 64
        d_th = np.pi * 2 / n_search
        xs, ys = [], []
        for i in range(n_search):
            th = i * d_th
            xs.append(np.cos(th))
            ys.append(np.sin(th))

        xs = np.array(xs)
        ys = np.array(ys)

        sf = 0.8
        nvecs = []
        widths = []
        for i in range(self.N):
            pt = self.cline[i]
            c, r = self.xy_to_row_column(pt)
            val = self.dt[r, c] * sf 
            widths.append(val)

            s_vals = np.zeros(n_search)
            s_pts = np.zeros((n_search, 2))
            for j in range(n_search):
                dpt = np.array([xs[j]+val, ys[j]*val]) / self.resolution
                # dpt_c, dpt_r = self.xy_to_row_column(dpt)
                # s_vals[i] = self.dt[r+dpt_r, c+dpt_c]
                s_pt = [int(round(r+dpt[1])), int(round(c+dpt[0]))]
                s_pts[j] = s_pt
                s_vals[j] = self.dt[s_pt[0], s_pt[1]]

            print(f"S_vals: {s_vals}")
            idx = np.argmin(s_vals) # closest to border

            th = d_th * idx

            nvec = [xs[idx], ys[idx]]
            nvecs.append(nvec)

            self.plot_nvec_finding(nvecs, widths, s_pts, pt)

        self.nvecs = np.array(nvecs)
        plt.show()

    def find_nvecs_old(self):
        N = self.N
        track = self.cline

        nvecs = []
        # new_track.append(track[0, :])
        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[0, :], track[1, :]))
        nvecs.append(nvec)
        for i in range(1, len(track)-1):
            pt1 = track[i-1]
            pt2 = track[min((i, N)), :]
            pt3 = track[min((i+1, N-1)), :]

            th1 = lib.get_bearing(pt1, pt2)
            th2 = lib.get_bearing(pt2, pt3)
            if th1 == th2:
                th = th1
            else:
                dth = lib.sub_angles_complex(th1, th2) / 2
                th = lib.add_angles_complex(th2, dth)

            new_th = th + np.pi/2
            nvec = lib.theta_to_xy(new_th)
            nvecs.append(nvec)

        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[-2, :], track[-1, :]))
        nvecs.append(nvec)

        self.nvecs = np.array(nvecs)

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def plot_nvec_finding(self, nvecs, widths, s_pts, c_pt, wait=False):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, origin='lower')

        xs, ys = [], []
        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            xs.append(s_x)
            ys.append(s_y)
        plt.plot(xs, ys, linewidth=2)

        for i in range(len(s_pts)-1):
            plt.plot(s_pts[i, 1], s_pts[i, 0], 'x')

        c, r = self.xy_to_row_column(c_pt)
        plt.plot(c, r, '+', markersize=20)

        for i in range(len(nvecs)):
            pt = self.cline[i]
            n = nvecs[i]
            w = widths[i]
            dpt = np.array([n[0]*w, n[1]*w])
            p1 = pt - dpt
            p2 = pt + dpt

            lx, ly = self.convert_positions(np.array([p1, p2]))
            plt.plot(lx, ly, linewidth=1)

            # plt.plot(p1, p2)
        plt.pause(0.001)


        cx, cy = self.convert_positions(self.cline)
        plt.plot(cx, cy, '--', linewidth=2)

        # plt.show()
        plt.pause(0.0001)
        if wait:
            plt.show()

    def set_true_widths(self):
        tx = self.cline[:, 0]
        ty = self.cline[:, 1]

        sf = 1 # safety factor
        nws, pws = [], []

        for i in range(self.N):
            pt = [tx[i], ty[i]]
            c, r = self.xy_to_row_column(pt)
            val = self.dt[r, c] * sf
            nws.append(val)
            pws.append(val)

        nws, pws = np.array(nws), np.array(pws)

        self.widths =  np.concatenate([nws[:, None], pws[:, None]], axis=-1)     
        # self.widths *= 0.2 #TODO: remove
        # self.widths *= 0.6 #TODO: remove

    def render_map(self, wait=False):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, origin='lower')


        ns = self.nvecs 
        ws = self.widths
        l_line = self.cline - np.array([ns[:, 0] * ws[:, 0], ns[:, 1] * ws[:, 0]]).T
        r_line = self.cline + np.array([ns[:, 0] * ws[:, 1], ns[:, 1] * ws[:, 1]]).T


        cx, cy = self.convert_positions(self.cline)
        plt.plot(cx, cy, '--', linewidth=2)
        lx, ly = self.convert_positions(l_line)
        plt.plot(lx, ly, linewidth=1)
        rx, ry = self.convert_positions(r_line)
        plt.plot(rx, ry, linewidth=1)

        for i, pt in enumerate(self.cline):
            plt.plot([lx[i], rx[i]], [ly[i], ry[i]])

        # if self.wpts is not None:
        #     wpt_x, wpt_y = self.convert_positions(self.wpts)
        #     plt.plot(wpt_x, wpt_y, linewidth=2)

        # plt.show()
        plt.pause(0.0001)
        if wait:
            plt.show()

    def check_scan_location(self, pt):
        c, r = self.xy_to_row_column(pt)
        if abs(c) > self.width -2 or abs(r) > self.height -2:
            return True
        val = self.dt[c, r]
        if val < 0.05:
            return True
        return False

    def save_map_std(self):
        filename = 'map_data/' + self.map_name + '_std.csv'

        track = np.concatenate([self.cline, self.nvecs, self.widths], axis=-1)

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Saved in File: {filename}")

def lines_cross(pt_1_l, pt_1_r, pt_2_l, pt_2_r):
    p1 = Point(pt_1_l[0], pt_1_l[1])
    p2 = Point(pt_1_r[0], pt_1_r[1])
    q1 = Point(pt_2_l[0], pt_2_l[1])
    q2 = Point(pt_2_r[0], pt_2_r[1])

    return doIntersect(p1,p2, q1, q2)
 
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False
 
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
     
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.
     
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
         
        # Clockwise orientation
        return 1
    elif (val < 0):
         
        # Counterclockwise orientation
        return 2
    else:
         
        # Collinear orientation
        return 0
 
# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
     
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
 
    # Special Cases
 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
 
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
 
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
 
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
 
    # If none of the cases
    return False

def run_pre_map():
    fname = "config_file"
    conf = lib.load_conf(fname)
    # map_name = "porto"
    # map_name = "berlin"
    # map_name = "race_track"
    
    map_name = "f1_aut_wide"
    # map_name = "example_map"
    # map_name = "levine_blocked"
    # map_name = "columbia_small"

    pre_map = ProcessMap(conf, map_name)
    pre_map.run_conversion()
    # pre_map.run_opti()


if __name__ == "__main__":
    run_pre_map()

