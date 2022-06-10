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


class OptimiseMap:
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

    def run_opti(self):
        self.read_yaml_file()
        self.load_map()
        self.load_track_pts()

        self.dt = ndimage.distance_transform_edt(self.map_img) 
        self.dt = np.array(self.dt *self.resolution)

        width_sf = 0.7
        n_set = MinCurvatureTrajectory(self.cline, self.nvecs, self.widths*width_sf)

        deviation = np.array([self.nvecs[:, 0] * n_set[:, 0], self.nvecs[:, 1] * n_set[:, 0]]).T
        self.wpts = self.cline + deviation

        # self.vs = Max_velocity(self.wpts, self.conf, False)
        self.vs = Max_velocity(self.wpts, self.conf, True)

        # plt.figure(4)
        # plt.plot(self.vs)

        self.save_map_opti()
        self.render_map(True)

    def load_track_pts(self):
        track = []
        filename = 'map_data/' + self.map_name + "_std.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename} in env_map")

        self.N = len(track)
        self.cline = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.widths = track[:, 4:6]
        
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


    def render_map(self, wait=False):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, origin='lower')

        xs, ys = [], []
        for pt in self.cline:
            s_x, s_y = self.xy_to_row_column(pt)
            xs.append(s_x)
            ys.append(s_y)
        plt.plot(xs, ys, linewidth=2)

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

        if self.wpts is not None:
            wpt_x, wpt_y = self.convert_positions(self.wpts)
            plt.plot(wpt_x, wpt_y, linewidth=2)

        # plt.show()
        plt.pause(0.0001)
        if wait:
            plt.show()

    def save_map_opti(self):
        filename = 'map_data/' + self.map_name + '_opti.csv'

        dss, ths = convert_pts_s_th(self.wpts)
        ss = np.cumsum(dss)
        ks = np.zeros_like(ths[:, None]) #TODO: add the curvature

        track = np.concatenate([ss[:, None], self.wpts[:-1], ths[:, None], ks, self.vs], axis=-1)

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(track)

        print(f"Track Saved in File: {filename}")
    
    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        if c >= self.dt.shape[1]:
            c = self.dt.shape[1] - 1
        if r >= self.dt.shape[0]:
            r = self.dt.shape[0] - 1

        return c, r

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

def MinCurvatureTrajectory(pts, nvecs, ws):
    """
    This function uses optimisation to minimise the curvature of the path
    """
    w_min = - ws[:, 0] 
    w_max = ws[:, 1] 
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

    N = len(pts)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [pts[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [pts[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [pts[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [pts[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # objective
    real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
    im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

    sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N-1)

    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])), 
    # 'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(pts[i, 0:2], pts[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    # thetas = np.array(x_opt[1*N:2*(N-1)])

    return n_set



"""Find the max velocity """
def Max_velocity(pts, conf, show=False):
    mu = conf.mu
    m = conf.m
    g = conf.g
    l_f = conf.l_f
    l_r = conf.l_r
    f_max = mu * m * g 
    f_long_max = l_f / (l_r + l_f) * f_max
    max_v = conf.max_v  
    max_a = conf.max_a

    s_i, th_i = convert_pts_s_th(pts)
    th_i_1 = th_i[:-1]
    s_i_1 = s_i[:-1]
    N = len(s_i)
    N1 = len(s_i) - 1

    # setup possible casadi functions
    d_x = ca.MX.sym('d_x', N-1)
    d_y = ca.MX.sym('d_y', N-1)
    vel = ca.Function('vel', [d_x, d_y], [ca.sqrt(ca.power(d_x, 2) + ca.power(d_y, 2))])
    # f_total = ca.Function('vel', [d_x, d_y], [ca.sqrt(ca.power(d_x, 2) + ca.power(d_y, 2))])

    dx = ca.MX.sym('dx', N)
    dy = ca.MX.sym('dy', N)
    dt = ca.MX.sym('t', N-1)
    f_long = ca.MX.sym('f_long', N-1)
    f_lat = ca.MX.sym('f_lat', N-1)

    nlp = {\
        'x': ca.vertcat(dx, dy, dt, f_long, f_lat),
        # 'f': ca.sum1(dt), 
        'f': ca.sumsqr(dt), 
        'g': ca.vertcat(
                    # dynamic constraints
                    dt - s_i_1 / ((vel(dx[:-1], dy[:-1]) + vel(dx[1:], dy[1:])) / 2 ),
                    # ca.arctan2(dy, dx) - th_i,
                    dx/dy - ca.tan(th_i),
                    dx[1:] - (dx[:-1] + (ca.sin(th_i_1) * f_long + ca.cos(th_i_1) * f_lat) * dt  / m),
                    dy[1:] - (dy[:-1] + (ca.cos(th_i_1) * f_long - ca.sin(th_i_1) * f_lat) * dt  / m),

                    # path constraints
                    ca.sqrt(ca.power(f_long, 2) + ca.power(f_lat, 2)),

                    # boundary constraints
                    vel(dx[:-1], dy[:-1]),
                    # vel(f_long, f_lat) # gets total force
                    # dx[0], dy[0]
                ) \
    }

    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})
    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})

    # make init sol
    v0 = np.ones(N) * max_v/2
    dx0 = v0 * np.sin(th_i)
    dy0 = v0 * np.cos(th_i)
    dt0 = s_i_1 / ca.sqrt(ca.power(dx0[:-1], 2) + ca.power(dy0[:-1], 2)) 
    f_long0 = np.zeros(N-1)
    ddx0 = dx0[1:] - dx0[:-1]
    ddy0 = dy0[1:] - dy0[:-1]
    a0 = (ddx0**2 + ddy0**2)**0.5 
    f_lat0 = a0 * m

    x0 = ca.vertcat(dx0, dy0, dt0, f_long0, f_lat0)

    # make lbx, ubx
    # lbx = [-max_v] * N + [-max_v] * N + [0] * N1 + [-f_long_max] * N1 + [-f_max] * N1
    lbx = [-max_v] * N + [0] * N + [0] * N1 + [-ca.inf] * N1 + [-f_max] * N1
    ubx = [max_v] * N + [max_v] * N + [10] * N1 + [ca.inf] * N1 + [f_max] * N1
    # lbx = [-max_v] * N + [0] * N + [0] * N1 + [-f_long_max] * N1 + [-f_max] * N1
    # ubx = [max_v] * N + [max_v] * N + [10] * N1 + [f_long_max] * N1 + [f_max] * N1

    #make lbg, ubg
    lbg = [0] * N1 + [0] * N + [0] * 2 * N1 + [0] * N1 + [0] * N1 #+ [-f_max] * N1 #+ [0] * 2 
    ubg = [0] * N1 + [0] * N + [0] * 2 * N1 + [ca.inf] * N1 + [max_v] * N1 #+ [f_max] * N1  #+ [0] * 2 

    r = S(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    dx = np.array(x_opt[:N])
    dy = np.array(x_opt[N:N*2])
    dt = np.array(x_opt[2*N:N*2 + N1])
    f_long = np.array(x_opt[2*N+N1:2*N + N1*2])
    f_lat = np.array(x_opt[-N1:])

    f_t = (f_long**2 + f_lat**2)**0.5

    # print(f"Dt: {dt.T}")
    # print(f"DT0: {dt[0]}")
    t = np.cumsum(dt)
    t = np.insert(t, 0, 0)
    print(f"Total Time: {t[-1]}")
    # print(f"Dt: {dt.T}")
    # print(f"Dx: {dx.T}")
    # print(f"Dy: {dy.T}")

    vs = (dx**2 + dy**2)**0.5
    vs = np.clip(vs, 0, max_v)

    if show:
        plt.figure(1)
        plt.clf()
        plt.title("Velocity vs dt")
        plt.plot(t, vs)
        plt.plot(t, th_i)
        plt.legend(['vs', 'ths'])
        # plt.plot(t, dx)
        # plt.plot(t, dy)
        # plt.legend(['v', 'dx', 'dy'])
        plt.plot(t, np.ones_like(t) * max_v, '--')

        plt.figure(3)
        plt.clf()
        plt.title("F_long, F_lat vs t")
        plt.plot(t[:-1], f_long)
        plt.plot(t[:-1], f_lat)
        plt.plot(t[:-1], f_t, linewidth=3)
        # plt.ylim([-25, 25])
        plt.plot(t, np.ones_like(t) * f_max, '--')
        plt.plot(t, np.ones_like(t) * -f_max, '--')
        plt.plot(t, np.ones_like(t) * f_long_max, '--')
        plt.plot(t, np.ones_like(t) * -f_long_max, '--')

        plt.legend(['Flong', "f_lat", "f_t"])

        # plt.show()
        plt.pause(0.001)
    
    # plt.figure(9)
    # plt.clf()
    # plt.title("F_long, F_lat vs t")
    # plt.plot(t[:-1], f_long)
    # plt.plot(t[:-1], f_lat)
    # plt.plot(t[:-1], f_t, linewidth=3)
    # plt.plot(t, np.ones_like(t) * f_max, '--')
    # plt.plot(t, np.ones_like(t) * -f_max, '--')
    # plt.plot(t, np.ones_like(t) * f_long_max, '--')
    # plt.plot(t, np.ones_like(t) * -f_long_max, '--')
    # plt.legend(['Flong', "f_lat", "f_t"])


    return vs


def convert_pts_s_th(pts):
    N = len(pts)
    s_i = np.zeros(N-1)
    th_i = np.zeros(N-1)
    for i in range(N-1):
        s_i[i] = lib.get_distance(pts[i], pts[i+1])
        th_i[i] = lib.get_bearing(pts[i], pts[i+1])

    return s_i, th_i



def run_opti():
    fname = "config_file"
    conf = lib.load_conf(fname)
    
    # map_name = "levine_blocked"
    # map_name = "columbia_small"
    # map_name = "example_map"
    map_name = "f1_aut_wide"

    pre_map = OptimiseMap(conf, map_name)
    pre_map.run_opti()


if __name__ == "__main__":
    run_opti()