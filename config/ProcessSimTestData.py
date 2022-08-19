from ctypes import alignment
import numpy as np
from matplotlib import pyplot as plt
import csv, yaml
from PIL import Image
from RacingRewards.DataTools.MapData import MapData
import glob
import trajectory_planning_helpers as tph
from RacingRewards.Reward import RaceTrack 
from matplotlib.ticker import PercentFormatter

class LapHistory:
    def __init__(self, vehicle_name, map_name):
        self.path = "Data/Vehicles/" + vehicle_name + "/"
        self.vehicle_name = vehicle_name
        self.states = None
        self.actions = None
        self.map_data = MapData(map_name)
        self.race_track = RaceTrack(map_name)
        self.map_name = map_name
        self.race_track.load_centerline()
        self.lap_n = 0

    def generate_path_data(self, lap_n):
        self.lap_n = lap_n
        try:
            map_name = "levine_blocked"
            data = np.load(self.path + f"Lap_{lap_n}_{map_name}_history_{self.vehicle_name}.npy")
            # data = np.load(self.path + f"Lap_{lap_n}_history_{self.vehicle_name}.npy")
        except:
            print(f"No data for lap {lap_n}")
            return
        self.states = data[:, :5]
        self.actions = data[:, 5:]

        steering = np.abs(self.actions[:, 0])
        mean_steering = np.mean(np.abs(steering)) *0.8

        pts = self.states[2:, 0:2]
        ss = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_distance = np.sum(ss)

        ths, ks = tph.calc_head_curv_num.calc_head_curv_num(pts, ss, False)
        mean_curvature = np.mean(np.abs(ks))
        total_curvature = np.sum(np.abs(ks))

        hs = []
        for point in pts:
            idx, dists = self.race_track.get_trackline_segment(point)
            x, h = self.race_track.interp_pts(idx, dists)
            hs.append(h)

        hs = np.array(hs)
        mean_deviation = np.mean(hs)
        total_deviation = np.sum(hs)

        data = {}
        name = self.vehicle_name.split("_")[0]
        if name != "PP": name = name.replace("Bench", "")
        data["name"] = name
        data["mean_steering"] = mean_steering
        data["total_distance"] = total_distance
        data["mean_curvature"] = mean_curvature
        data["total_curvature"] = total_curvature
        data["mean_deviation"] = mean_deviation
        data["total_deviation"] = total_deviation

        return data 


def generate_result_table():
    map_names = ["levine_blocked"]
    # map_names = ["lobby"]
    planner_names = ['experiment_2', "BaselineSSS_levine_blocked_2_0_0"]
    for map_name in map_names:
        ds = []
        for name in planner_names:
            # folder = f"Data/Vehicles/{name}/"
            # print(folder)
            lap_history = LapHistory(name, map_name)
            data = lap_history.generate_path_data(1)

            ds.append(data)

        with open(f"Data/PaperData/sim_{map_name}_results.txt", "w") as f:
            names = ["\\textbf{" + ds[i]['name'] + "}" for i in range(len(ds))]
            mean_steerings = ["%10.3f" % ds[i]['mean_steering'] for i in range(len(ds))]
            total_distances = ["%10.1f $m$" % ds[i]['total_distance'] for i in range(len(ds))]
            mean_curvatures = ["%10.2f" % ds[i]['mean_curvature'] for i in range(len(ds))]
            total_curvatures = ["%10.1f" % ds[i]['total_curvature'] for i in range(len(ds))]
            total_deviation = ["%10.1f" % ds[i]['total_deviation'] for i in range(len(ds))]
            mean_deviation = ["%10.2f" % ds[i]['mean_deviation'] for i in range(len(ds))]

            name_str = " & ".join(names) 
            mean_steerings = " & ".join(mean_steerings)
            mean_curvature_str = " & ".join(mean_curvatures)
            total_curvature_str = " & ".join(total_curvatures)
            distance_str = " & ".join(total_distances)
            mean_deviation_str = " & ".join(mean_deviation)
            total_deviation_str = " & ".join(total_deviation)
            f.write(f"\\textbf{{Metric}} & {name_str} \\\\ \n")
            f.write(f"Path Distance & {distance_str} \\\\ \n")
            f.write(f"Mean Curvature & {mean_curvature_str} \\\\ \n")
            f.write(f"Total Curvature & {total_curvature_str} \\\\ \n")
            f.write(f"Mean Steering & {mean_steerings} \\\\ \n")
            f.write(f"Mean Deviation & {mean_deviation_str} \\\\ \n")
            f.write(f"Total Deviation & {total_deviation_str} \\\\ \n")



        


if __name__ == '__main__':
    generate_result_table()