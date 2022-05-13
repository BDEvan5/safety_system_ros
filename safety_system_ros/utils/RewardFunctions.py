
import numpy as np 
import csv



def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)

def find_closest_pt(pt, wpts):
    """
    Returns the two closes points in order along wpts
    """
    dists = [get_distance(pt, wpt) for wpt in wpts]
    min_i = np.argmin(dists)
    d_i = dists[min_i] 
    if min_i == len(dists) - 1:
        min_i -= 1
    if dists[max(min_i -1, 0) ] > dists[min_i+1]:
        p_i = wpts[min_i]
        p_ii = wpts[min_i+1]
        d_i = dists[min_i] 
        d_ii = dists[min_i+1] 
    else:
        p_i = wpts[min_i-1]
        p_ii = wpts[min_i]
        d_i = dists[min_i-1] 
        d_ii = dists[min_i] 

    return p_i, p_ii, d_i, d_ii

def get_tiangle_h(a, b, c):
    s = (a + b+ c) / 2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    h = 2 * A / c

    return h

def distance_potential(s, s_p, end, beta=0.2, scale=0.5):
    prev_dist = get_distance(s[0:2], end)
    cur_dist = get_distance(s_p[0:2], end)
    d_dis = (prev_dist - cur_dist) / scale

    return d_dis * beta

def find_reward(s_p):
    if s_p['collisions'][0] == 1:
        return -1
    elif s_p['lap_counts'][0] == 1:
        return 1
    return 0

# Track base
class TrackPtsBase:
    def __init__(self, config) -> None:
        self.wpts = None
        self.ss = None
        self.map_name = config.map_name
        self.directory = config.directory
        self.total_s = None

    def load_center_pts(self):
        track_data = []
        filename = f"{self.directory}map_data/{self.map_name}_centerline.csv"
        
        try:
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
                for lines in csvFile:  
                    track_data.append(lines)
        except FileNotFoundError:
            raise FileNotFoundError("No map file center pts")

        track = np.array(track_data)
        print(f"Track Loaded: {filename} in reward")

        N = len(track)
        self.wpts = track[:, 0:2]
        ss = np.array([get_distance(self.wpts[i], self.wpts[i+1]) for i in range(N-1)])
        ss = np.cumsum(ss)
        self.ss = np.insert(ss, 0, 0)

        self.total_s = self.ss[-1]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def load_reference_pts(self):
        track_data = []
        filename = f"{self.directory}map_data/{self.map_name}_opti.csv"
        
        try:
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
                for lines in csvFile:  
                    track_data.append(lines)
        except FileNotFoundError:
            raise FileNotFoundError("No reference path")

        track = np.array(track_data)
        print(f"Track Loaded: {filename} in reward")

        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]

        self.total_s = self.ss[-1]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def find_s(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        dist_from_cur_pt = dists[min_dist_segment]
        if dist_from_cur_pt > 1: #more than 2m from centerline
            return self.ss[min_dist_segment] - dist_from_cur_pt # big makes it go back

        s = self.ss[min_dist_segment] + dist_from_cur_pt

        return s 

    def get_distance_r(self, pt1, pt2, beta):
        s = self.find_s(pt1)
        ss = self.find_s(pt2)
        ds = ss - s
        scale_ds = ds / self.total_s
        r = scale_ds * beta
        shaped_r = np.clip(r, -0.5, 0.5)

        return shaped_r


class RefDistanceReward(TrackPtsBase):
    def __init__(self, config) -> None:
        TrackPtsBase.__init__(self, config)

        # self.load_reference_pts()
        self.load_center_pts()
        self.b_distance = 1

    def __call__(self, state, s_prime):
        prime_pos = np.array([s_prime['state'][0:2]])
        pos = np.array([state['state'][0:2]])

        reward = self.get_distance_r(pos, prime_pos, 1)

        reward += s_prime['reward']

        return reward

class RefCTHReward(TrackPtsBase):
    def __init__(self, conf) -> None:
        TrackPtsBase.__init__(self, conf)
        self.max_v = conf.max_v

        self.load_center_pts()
        # self.load_reference_pts()
        self.mh = conf.r1
        self.md = conf.r2
        self.rk = conf.rk


    def __call__(self, state, s_prime):
        # s_prime['reward'] = find_reward(s_prime)
        prime_pos = np.array([s_prime['poses_x'][0], s_prime['poses_y'][0]])
        theta = - s_prime['poses_theta'][0] + np.pi/2 #! NOTE: get angle right for simulator
        velocity = s_prime['linear_vels_x'][0]

        pt_i, pt_ii, d_i, d_ii = find_closest_pt(prime_pos, self.wpts)
        d = get_distance(pt_i, pt_ii)
        d_c = get_tiangle_h(d_i, d_ii, d) 

        th_ref = get_bearing(pt_i, pt_ii)
        th = theta
        d_th = abs(sub_angles_complex(th_ref, th))
        v_scale = velocity / self.max_v

        r_h = self.mh * np.cos(d_th) * v_scale
        r_d = self.md * d_c
        new_r = r_h - r_d - self.rk

        return new_r + s_prime['reward']




def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)
     
def get_gradient(x1=[0, 0], x2=[0, 0]):
    t = (x1[1] - x2[1])
    b = (x1[0] - x2[0])
    if b != 0:
        return t / b
    return 1000000 # near infinite gradient. 


def get_bearing(x1=[0, 0], x2=[0, 0]):
    grad = get_gradient(x1, x2)
    dx = x2[0] - x1[0]
    th_start_end = np.arctan(grad)
    if dx == 0:
        if x2[1] - x1[1] > 0:
            th_start_end = 0
        else:
            th_start_end = np.pi
    elif th_start_end > 0:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi/2 - th_start_end
    else:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = - np.pi/2 - th_start_end

    return th_start_end

import math, cmath
def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase
    
