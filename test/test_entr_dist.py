import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.exposure import histogram
from sklearn.metrics import mutual_info_score
# from scipy.misc import imread
import numpy as np
import cv2
# from scipy.stats import entropy
from tutils import torchvision_save
from einops import rearrange


def draw_bar_auto_split(values, tag=0, draw_guassian_line=False):
    thresholds = np.arange(np.floor(values.min()), np.ceil(values.max()), 0.2).astype(float)
    print()
    thresholds[0] = 0
    thresholds[-1] = 999
    # print(thresholds, thresholds.shape)
    mre = np.array(values)
    print(mre.shape)
    # print(mre)
    length_collect = []
    pre_len = 0
    for i in range(len(thresholds)-1):
        ind = np.where(mre<=thresholds[i+1] )[0]
        length_collect.append(len(ind) - pre_len)
        pre_len = len(ind)
        # print("mmm: ", len(ind), length_collect)
    length_collect = np.array(length_collect) / len(mre)
    thresholds_str = [f"{i:.1f}" for i in thresholds]
    print(thresholds_str)

    x_test = None
    y = None
    if draw_guassian_line:
        mean = mre.mean()
        std = mre.std()
        x_test = np.linspace(0, 3, 100)
        print("????", mean, std)
        def _gaussian(x, mean, std):
            a = 1 / np.sqrt(2 * 3.141592 * std ** 2)
            y = a * np.exp(-(x-mean)**2 / (2 * std**2))
            return y
        y = [_gaussian(x, mean, std) for x in x_test]
        y = np.array(y)
        print(y)
        print(x_test)
        # plt.plot(x, y)
    draw_bar(thresholds_str, length_collect, fname=f"entr_debug_{tag}.png", color="blue", xlabel="Mutual Information (MI)", ylabel="Percentage (%)", ex_x=x_test, ex_y=y)
    return thresholds_str, length_collect


def draw_bar(labels, values, fname="tbar.pdf", title=None, color="red", set_font=None, xlabel="x", ylabel="y", ex_x=None, ex_y=None):
    plt.ioff()
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    # fig = plt.figure(figsize=(11,6))
    fig, ax = plt.subplots(figsize=(20,12))
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
    plt.xticks(x_pos[:-1], labels[:-1], fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    if ex_x is not None and ex_y is not None:
        ax.plot(ex_x, ex_y, color="green", label="gaussian")
    plt.savefig(fname)
    plt.close()
    print("Drawed img: ", fname)


pth = "/home1/quanquan/datasets/Cephalometric/entr1/train/1.npy"
data = np.load(pth)
data = rearrange(data, "h w-> (h w)")
draw_bar_auto_split(data)
