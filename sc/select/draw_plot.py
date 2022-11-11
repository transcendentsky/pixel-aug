import numpy as np
import argparse
from tutils import trans_init, trans_args, dump_yaml, save_script


import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
def draw_scatter(points, points2, fname="ttest.png", c="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()  # Turn off interactive plotting off
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    fig = plt.figure()
    points = points.flatten()
    points2 = points2.flatten()
    plt.scatter(points, points2, c=c, label="???")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()

data_inds = [5,10, 15, 20, 30]
data_max = [2.56, 2.41, 2.33, 2.25, 2.11]
data_min = [2.37, 2.24, 2.122, 2.06, 2.04, ]
data_our = [2.37, 2.247, 2.12592, 2.067, 2.055]

plt.plot(data_inds, data_our, c='red')
plt.fill_between(data_inds, data_min, data_max, facecolor='yellow')
plt.savefig("test.png")