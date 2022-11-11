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
    labels = [str(label) for label in labels]
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
    # thresholds[-1] = 999
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



if __name__ == '__main__':
    # draw_bar_maxsim()
    draw_bar_sim_n1()

