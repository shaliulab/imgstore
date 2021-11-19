"""
Module to read and analyze the environmental metadata in .extra.json files
"""

import argparse
import os.path
import rle
import datetime

import imgstore
from shapely.geometry import Polygon
from descartes import PolygonPatch


import numpy as np
import matplotlib.pyplot as plt


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add.argument("--experiment-path", "--input", dest="input", required=True)
    ap.add.argument("--output", dest="output", required=True)
    return ap


def clean_data(data):
    """
    Remove rows where no environmental information is available
    """
    data=data.loc[~np.isnan(data["humidity"])]
    data=data.loc[~np.isnan(data["temperature"])]
    data.reset_index(inplace=True)
    return data


def align_data_to_zt0(data, store_datetime_str):
    """
    Generate a new column called ZT0 that states ms since ZT0
    """
    # store_datetime_str = re.findall("([\d]{4}-[\d]{2}-[\d]{2}_[\d]{2}-[\d]{2}-[\d]{2})", store_path)[0]
    store_datetime = datetime.datetime.strptime(store_datetime_str,"%Y-%m-%d_%H-%M-%S")
    zt0 = datetime.datetime(year=store_datetime.year, month=store_datetime.month, day=store_datetime.day, hour=7)
    print(f"Start time detected {store_datetime}")
    print(f"ZT0 detected {zt0}")
    offset = (store_datetime - zt0)
    offset_ms = offset.seconds * 1000
    data["ZT"] = data["frame_time"] + offset_ms
    return data


def discretize_light(data):
    """
    The photoresistor does not output binary, but a continuum of light, with two main modes
    which represent the D and L phase.
    So all data above the mean belongs to the L phase
    and below is the D phase
    This information is stored in the L column
    """
    threshold = data["light"].mean()
    print(threshold)
    data["L"] = [str(e)[0] for e in data["light"] > threshold]
    return data


def geom_ld_annotations(data, ax):

    values, accum = rle.get_rle(data["L"].values)

    zts = []
    pos=[data.index[0]]
    pos.extend(accum[:-1])

    for i in pos:
        zts.append(round(data.loc[i]["t"], 2))

    max_t = data["t"].tail().values[-1]
    min_t = zts[0]
    y_max  = 100
    color = {"F": (0, 0, 0), "T": (1, 1, 1)}
    for i in range(len(zts)):
        if (i + 1) == len(zts):
            ring_mixed = Polygon([(zts[i], 0), (max_t, 0), (max_t, y_max), (zts[i], y_max)])        
        else:
            ring_mixed = Polygon([(zts[i], 0), (zts[i+1], 0), (zts[i+1], y_max), (zts[i], y_max)])
        ring_patch = PolygonPatch(ring_mixed, facecolor=color[values[i]], alpha=0.2, edgecolor=(0,0,0))
        ax.add_patch(ring_patch)

    xrange = [int(np.floor(min_t)), int(np.ceil(max_t))]
    yrange = [0, y_max]
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    xticks = np.arange(6* (int(min_t) // 6), 6*(int(max_t + 6) // 6), 6)
    ax.set_xticks(xticks)

    ax.set_xlabel("ZT")
    
    return ax


def geom_env_data(data, ax):
    
    ax2 = ax.twinx()
    ax2.set_ylabel("Temp ºC")
    ax.set_ylabel("% Hum")
    ax.scatter(data["t"], data["humidity"], s=.1)
    ax2.scatter(data["t"], data["temperature"], c="red", s=.1)
    return ax, ax2


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    store_datetime_str =os.path.basename(args.input.strip("/"))
    store=imgstore.new_for_filename(args.input)
    env_data = store.get_extra_data(ignore_corrupt_chunks=True)

    env_data = clean_data(env_data)
    env_data = discretize_light(env_data)
    env_data["t"] = env_data["ZT"] / 3600000

    fig = plt.figure(1, figsize=(5,5), dpi=90)
    ax = fig.add_subplot(111)
    ax = geom_ld_annotations(env_data, ax)
    ax.set_title(store_datetime_str)
    geom_env_data(env_data, ax)
    plt.tight_layout()
    fig.savefig(args.output)
