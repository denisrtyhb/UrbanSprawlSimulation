# Import necessary packages

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os
import matplotlib.pyplot as plt
# Use geopandas for vector data and rasterio for raster data
import rasterio as rio
# Plotting extent is used to plot raster & vector data together
from rasterio.plot import plotting_extent

import numpy as np

import math

import random

import scipy

import sklearn.linear_model

from tqdm import tqdm

import config

from matplotlib.colors import LinearSegmentedColormap
colors = np.array([
    (173, 216, 230), # non-urban
    (160, 32,  240), # urban
    (0,   0,   139), # water
    (255, 255, 255), # background
]) / 255
cmap = LinearSegmentedColormap.from_list('my_list', colors)

def simulate_random(array, p_func):
    res = np.copy(array)
    prb = p_func(array)
    res[(res == 0) * (prb > np.random.rand(map_height, map_width))] = 1
    return res

def simulate_deter(array, p_func, threshold=0.5, t=1):
    res = np.copy(array)
    prb = p_func(array, t)
    res[(res == 0) * (prb > threshold)] = 1
    return res

def score(pred, before, after):
    ''' calculating an accuracy of the predicted development of 'before' map '''
    correctSim = np.sum((pred == after) * (after == 1) * (before == 0))
    missedSim = np.sum((after == 1) * (pred == 0) * (before == 0))
    falseSim = np.sum((after == 0) * (pred == 1) * (before == 0))
    print(f"correctSim: {correctSim}, missedSim: {missedSim}, falseSim: {falseSim}")
    return correctSim / (correctSim + missedSim + falseSim)

def P_neigh(array, LAP, neigh_radius):
    mask = np.ones((neigh_radius, neigh_radius))/(neigh_radius ** 2 - 1)
    mask[neigh_radius // 2][neigh_radius // 2] = 0
    return scipy.signal.convolve2d(array == 1, mask, mode="same")

def get_p_func(P_driver, LAP, neigh_radius, TIP):
    def P_total(array, t=1):
        return (P_neigh(array, LAP, neigh_radius) * LAP + P_driver() * (1 + TIP) ** (t - 1)) / 2
    return P_total

def save_result(name, arr):
    # saving as .tif
    with rio.Env():
        with rio.open(
            f'{config.results_folder}/{name}.tif',
            'w',
            driver='GTiff',
            count=1,
            dtype='uint8',
            compress='lzw',
            width=arr.shape[1],
            height=arr.shape[0],
            blockxsize=128,
            blockysize=128,
            tiled=True,
            interleave='band',
            nodata=3,
            transform=rio.Affine(30.0, 0.0, 469969.29371048906, 0.0, -30.0, 3600420.6514008716),
            crs=rio.CRS.from_epsg(32650),
        ) as dst:
            dst.write(arr.astype(rio.uint8), 1)

    # saving as jpg

    plt.imshow(arr, cmap=cmap)
    plt.title(name)
    plt.axis('off')
    plt.savefig(os.path.join(config.results_folder, name + ".jpg"))
    plt.show()
