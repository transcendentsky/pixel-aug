"""
    See jupyter notebook
"""
import numpy as np
from tutils import tfilename
import os
import numpy as np
from tutils import trans_init, trans_args, dump_yaml, tfilename, CSVLogger
import argparse
import seaborn as sns

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns


def draw_scatter(points, points2, fname="ttest.pdf", title=None, c="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()  # Turn off interactive plotting off
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    fig = plt.figure()
    parent, tail = os.path.split(fname)
    if title is not None:
        fig.suptitle(title)
    points = points.flatten()
    points2 = points2.flatten()
    plt.scatter(points, points2, c=c, label="???")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()


def seaborn_scatter(x, y, fname="ttest.pdf", xlabel="x", ylabel="y", color="blue", ex_points=None):
    sns.set_theme(style="whitegrid", font='Times New Roman', font_scale=1.2)
    # fig = sns.scatterplot(x=x, y=y, color=color)
    # fig = sns.regplot(x=x, y=y, color=color, line_kws={'color': "cyan", 'alpha': 0.5}, order=2)
    # fig = sns.scatterplot(x=x, y=y)
    fig = sns.lineplot(x=x, y=y, label="GU$\mathregular{^2}$Net")
    x2, y2 = ex_points
    fig = sns.scatterplot(x=x2, y=y2, color="red", label="CC2D")
    fig.text(25.71,2.78, "T=1", horizontalalignment='left', size='medium', color='black', weight='semibold')
    fig.text(31.5,2.43, "T=5", horizontalalignment='left', size='medium', color='black', weight='semibold')
    fig.text(34.05,2.24, "T=10", horizontalalignment='left', size='medium', color='black', weight='semibold')
    scatter_fig = fig.get_figure()
    fig.set_xlim(15,38)
    fig.set_ylim(1.5, 5)
    fig.set(xlabel=xlabel, ylabel=ylabel)
    plt.xticks([14,16,18,20,22,24,26,28,30,32,34,36,38])
    # plt.legend()
    scatter_fig.savefig(fname+".png", dpi=400)
    scatter_fig.savefig(fname+".pdf", dpi=400)
    print("Save to: ", fname)



if __name__ == "__main__":
    import csv
    alist = []
    alist_x = []
    alist_y = []
    with open("ceph.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            # print(row)
            xx = int(row[0])
            for rr in row[1:]:
                yy = float(rr)
                alist.append([xx, yy])
                alist_x.append(xx)
                alist_y.append(yy)
    # alist_x.append(4.0)
    # alist_y.append(41.0)
    print(alist_x, alist_y)
    ex_points = [[26.50, 32.07,33.88], [2.60, 2.30, 2.24]]
    seaborn_scatter(alist_x, alist_y,
                    fname="ttest",
                    ex_points=ex_points,
                    xlabel="Num of labeled images",
                    ylabel="MRE")
