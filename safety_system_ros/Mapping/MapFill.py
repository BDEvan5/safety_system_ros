import numpy as np
from matplotlib import pyplot as plt
import yaml, csv
from PIL import Image 
import sys

import faulthandler
faulthandler.enable()

def MapFiller(map_name, pts, crop_x, crop_y):

    file_name = 'maps/' + map_name + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)
        yaml_file = dict(documents.items())

    try:
        resolution = yaml_file['resolution']
        origin = yaml_file['origin']
        map_img_path = 'maps/' + yaml_file['image']
    except Exception as e:
        print(f"Problem loading, check key: {e}")
        raise FileNotFoundError("Problem loading map yaml file")

    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    map_img = map_img.astype(np.float64)
    if len(map_img.shape) == 3:
        map_img = map_img[:, :, 0]

    map_img[map_img <= 210] = 1.
    # map_img[map_img <= 128.] = 1.
    map_img[map_img > 128.] = 0.


    map_img = map_img.T
    crop = True
    if crop:
        map_img = map_img[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]

        map_img = Image.fromarray(map_img)
        resize = 1
        map_img = map_img.resize((int(map_img.size[0] * resize), int(map_img.size[1] * resize)))
        map_img = np.array(map_img).astype(np.float64)
        map_img[map_img > 0.40] = 1

        # plt.figure(1)
        # plt.imshow(map_img.T, origin='lower')
        # plt.show()


    for pt in pts:
        print(f"Pt: {pt}")
        map_img = boundary_fill(map_img, pt[0], pt[1], 1)
    
    # map_img[map_img!=5] = 10
    # map_img[map_img==5] = 0

    plt.show()
    plt.figure(1)
    plt.title("Supposed to be good")
    plt.imshow(map_img.T, origin='lower')

    plt.show()

    
    map_img[map_img <0.8] = 255
    map_img[map_img < 128] = 0
    img = Image.fromarray(map_img.T.astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)

    img.save('maps/' + map_name + '_filled.png')

    plt.figure(1)
    plt.imshow(map_img.T, origin='lower')

    plt.show()

def view_map(map_name):
    file_name = 'maps/' + map_name + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)
        yaml_file = dict(documents.items())
    map_img_path = 'maps/' + yaml_file['image']
    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    map_img = map_img.astype(np.float64)
    if len(map_img.shape) == 3:
        map_img = map_img[:, :, 0]

    map_img[map_img <= 210] = 1.
    # map_img[map_img <= 128.] = 1.
    map_img[map_img > 128.] = 0.

    plt.figure(1)
    plt.imshow(map_img, origin='lower')

    plt.show()


def boundary_fill(map_img, i, j, n, fill=2, boundary=1):
    if map_img[i, j] != boundary and map_img[i, j] != fill:
        map_img[i, j] = fill
        ni = n+1
        if ni > 10000:   return map_img # limit to 1000 recursions
        if i > 1:
            boundary_fill(map_img, i - 1, j, ni, fill, boundary)
        if i < map_img.shape[0] - 1:
            boundary_fill(map_img, i + 1, j, ni, fill, boundary)
        if j > 1:
            boundary_fill(map_img, i, j - 1, ni, fill, boundary)
        if j < map_img.shape[1] - 1:
            boundary_fill(map_img, i, j + 1, ni, fill, boundary)

    return map_img

def run_porto():
    crop_x = [50, 375]
    crop_y = [200, 320]
    pts = [[0, 0],
            [76, 65]]

    MapFiller('porto', pts, crop_x, crop_y)

def run_torino():
    view_map("torino")
    # crop_x = [480, 950]
    # crop_y = [250, 1220]
    crop_y = [120, 610]
    crop_x = [240, 480]
    # pts = [[0, 0]]
    pts = []

    MapFiller('torino', pts, crop_x, crop_y)


def run_berlin():
    view_map("berlin")

    crop_x = [80, 490]
    crop_y = [0, -1]
    pts = [[0, 0]]

    MapFiller('berlin', pts, crop_x, crop_y)


def run_racetrack():
    view_map("race_track")

    crop_x = [480, 1050]
    crop_y = [490, 940]
    pts = [[0, 0]]
    # pts = []

    MapFiller('race_track', pts, crop_x, crop_y)


def run_example_map():
    view_map("example_map")

    crop_x = [350, 1300]
    crop_y = [630, 1200]
    pts = [[0, 0], [40, 60]]
    # pts = []

    MapFiller('example_map', pts, crop_x, crop_y)



def run_circle():
    view_map("circle")

    crop_x = [830, 1270]
    crop_y = [610, 1040]
    pts = [[0, 0]]
    # pts = []

    MapFiller('circle', pts, crop_x, crop_y)



def run_columbia():
    view_map("columbia_small")

    crop_x = [780, 1460]
    crop_y = [640, 1130]
    pts = [[0, 0]]
    # pts = []

    MapFiller('columbia_small', pts, crop_x, crop_y)


def run_aut():
    view_map("f1_aut_wide")

    crop_x = [820, 1400]
    crop_y = [550, 1050]
    pts = [[0, 0]]
    # pts = []

    MapFiller('f1_aut_wide', pts, crop_x, crop_y)


def run_torino_small():
    view_map("torino_redraw_small")

    crop_x = [170, 650]
    crop_y = [180, 370]
    pts = [[0, 0]]
    # pts = []

    MapFiller('torino_redraw_small', pts, crop_x, crop_y)


def run_blackbox():
    view_map("blackbox1")

    crop_x = [430, 1060]
    crop_y = [750, 1550]
    pts = [[0, 0]]
    # pts = []

    MapFiller('blackbox1', pts, crop_x, crop_y)


def run_levine():
    # view_map("levine_blocked")

    crop_x = [700, 1270]
    crop_y = [950, 1250]
    # pts = [[0, 0], [80, 60], [50, 10], [50, 110], [220, 5], [18, 6], [206, 10], [20, 112]]
    pts = []
    # pts = [[70, 70]]
    # pts = [[45, 60]]
    pts = [[30, 40], [200, 45], [100, 150], [200, 100], [300, 220], [200, 280], [555, 136], [555, 25], [214, 186], [50, 288], [50, 26], [520, 25], [308, 190], [135, 190], [73, 182], [172, 186]]

    MapFiller('levine_blocked', pts, crop_x, crop_y)


if __name__ == '__main__':
    sys.setrecursionlimit(10000000)
    # run_porto()
    # run_torino()
    # run_berlin()
    # run_racetrack()
    run_example_map()

    # run_circle()
    # run_columbia()
    # run_aut()
    # run_torino_small()
    # run_blackbox()
    # run_levine()