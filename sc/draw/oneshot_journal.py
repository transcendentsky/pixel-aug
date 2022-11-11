"""
    See jupyter notebook
"""
import numpy as np
from tutils import tfilename
import os

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import csv
import seaborn as sns



def draw_bar(labels, values, fname="tbar.pdf", title=None, color="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    # fig = plt.figure(figsize=(11,6))
    fig, ax = plt.subplots(figsize=(14,8))
    if title is not None:
        fig.suptitle(title)
    # ax = fig.add_axes([0,0,1,1])
    assert len(labels) == len(values) + 1
    x_pos = [i for i, _ in enumerate(labels)]
    x_pos2 = np.array(x_pos[:-1])
    width = 0.5
    print(x_pos2)
    # import ipdb; ipdb.set_trace()
    ax.bar(x_pos2 + width, values, alpha=0.7, color=color)
    ax.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.5)

    fontsize_ticks = 22
    fontsize_label = 28
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    labels = [str(f"{label:0.1f}") for label in labels]
    print(x_pos, labels)
    plt.xticks(x_pos[:-1], labels[:-1], fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig(fname)
    plt.close()
    print("Drawed img: ", fname)


def draw_bar_sim_n1():
    # mre_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n1_1/mre_n1_list.npy"

    mre_list = []
    with open("./record.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        row = next(reader)
        print(row)
        # import ipdb; ipdb.set_trace()
        for row in reader:
            # print(row)
            xx = float(row[0])
            mre_list.append(xx)

    mre = np.array(mre_list)
    print("Mre info:")
    print(mre.min(), mre.max())
    print(mre.shape)
    # print(mre)
    thresholds = np.linspace(2.4, 5.4, num=16)
    thresholds[-1] = 999
    print(thresholds, thresholds.shape)
    length_collect = []
    pre_len = 0
    for i in range(len(thresholds)-1):
        ind = np.where(mre<=thresholds[i+1] )[0]
        length_collect.append(len(ind) - pre_len)
        pre_len = len(ind)
        # thresholds.append(f"{i*0.05:.2f}")
        print("mmm: ", len(ind), length_collect)
        # import ipdb; ipdb.set_trace()
    # thresholds.append(f"{(end+1)*0.05:.2f}")
    length_collect = np.array(length_collect) / 150
    print(thresholds, length_collect)
    draw_bar(thresholds, length_collect, fname=f"mre_dist.png", color="blue", xlabel="Mean Radial Error (MRE)", ylabel="Quantity (%)")
    draw_bar(thresholds, length_collect, fname=f"mre_dist.pdf", color="blue", xlabel="Mean Radial Error (MRE)", ylabel="Quantity (%)")


def draw_scatter(points, points2, fname="scatter",
                 title=None, c="blue", set_font=None,
                 xlabel="x", ylabel="y",
                 cc=None):
    plt.ioff()  # Turn off interactive plotting off
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    fig = plt.figure()
    parent, tail = os.path.split(fname)
    fig = sns.regplot(x=points, y=points2, color='blue', line_kws={'color': "cyan", 'alpha': 0.5})
    print("cc: ", cc)
    if cc is not None:
        fig.text(3.54, 3.55, f"cc={cc[0][1]:0.3f}",
                 horizontalalignment='left', size='medium', color='black', weight='semibold')
    if title is not None:
        fig.suptitle(title)
    points = points.flatten()
    points2 = points2.flatten()
    plt.scatter(points, points2, c=c, label="???")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname+".png")
    plt.savefig(fname+".pdf")
    plt.close()


def draw_plot():
    pass


def draw_correlation_graph():
    # mre_list = []
    # with open("./record.csv", "r") as f:
    #     reader = csv.reader(f, delimiter=",")
    #     row = next(reader)
    #     print(row)
    #     # import ipdb; ipdb.set_trace()
    #     for row in reader:
    #         # print(row)
    #         xx = float(row[0])
    #         mre_list.append(xx)

    tlist = []
    tlist += [
        # [0, 7.91, 7.64],
        [1, 3.21, 3.07],
        [2, 3.24, 3.17],
        [3, 3.45, 3.37],
        [4, 3.15, 2.92],
        [5, 3.31, 3.23],
        # [1,],
    ]
    # Old record
    # tlist += [
    #     [0, 4.04, 2.72],
    #     [0, 4.16, 2.85],
    #     [0, 4.28, 2.92],
    #     [0, 4.35, 2.98],
    #     [0, 4.65, 3.54],
    # ]
    tlist += [
        [0, 2.85, 2.47],
        [0, 2.71, 2.41],
        [0, 2.58, 2.36],
        [0, 2.82, 2.58],
        [0, 2.64, 2.56],
        [0, 3.15, 2.58],
        [0, 2.95, 2.58],
        [0, 3.82, 3.35],
        [0, 3.64, 3.02],
    ]
    mre_list1 = [a[1] for a in tlist]
    mre_list2 = [a[2] for a in tlist]
    mre_list1 = np.array(mre_list1)
    mre_list2 = np.array(mre_list2)

    cc = np.corrcoef(mre_list1, mre_list2)

    draw_scatter(mre_list1, mre_list2, fname="wo_semiSL", cc=cc,
                 xlabel="MRE of CC2D-SelfSL (mm)",
                 ylabel="MRE of CC2d-SemiSL (mm)")


if __name__ == '__main__':
    draw_bar_sim_n1()
    # draw_correlation_graph()
