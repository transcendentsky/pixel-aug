import cv2
import numpy as np
import cv2
import matplotlib
from tutils import tfilename
from PIL import Image
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def draw_bar_auto_split(values, tag=0, draw_guassian_line=False):
    print("min, max, interval", np.floor(values.min()).astype(int), np.ceil(values.max()).astype(int), ((values.max()-values.min())).round() / 10)
    thresholds = np.arange(np.floor(values.min()).astype(int), np.ceil(values.max()).astype(int), ((values.max()-values.min())).round() / 10).astype(float)
    # print()
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
    thresholds_str = [f"{i:.2f}" for i in thresholds]
    print(thresholds_str)

    x_test = None
    y = None
    if draw_guassian_line:
        mean = mre.mean()
        std = mre.std()
        x_test = np.linspace(0, 3, 100)
        print("debug ????", mean, std)
        def _gaussian(x, mean, std):
            a = 1 / np.sqrt(2 * 3.141592 * std ** 2)
            y = a * np.exp(-(x-mean)**2 / (2 * std**2))
            return y
        y = [_gaussian(x, mean, std) for x in x_test]
        y = np.array(y)
        print(y)
        print(x_test)
        # plt.plot(x, y)
    draw_bar(thresholds_str, length_collect, fname=f"tbar_mi_ceph_lm{tag}.png", color="blue", xlabel="Mutual Information (MI)", ylabel="Percentage (%)", ex_x=x_test, ex_y=y, save=False)

    # draw_bar(thresholds, length_collect, fname=f"tbar_mre_n1.pdf", color="blue", xlabel="Mean Radial Error (MRE)", ylabel="Quantity (%)")
    return thresholds_str, length_collect


def draw_bar(labels, values, fname="tbar.pdf", title=None, color="red", set_font=None, xlabel="x", ylabel="y", ex_x=None, ex_y=None, save=True):
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
    plt.xticks(x_pos[:-1], labels[:-1], fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    if ex_x is not None and ex_y is not None:
        ax.plot(ex_x, ex_y, color="green", label="gaussian")

    if save:
        plt.savefig(fname)
        plt.close()
        print("Drawed img: ", fname)
    else:
        return plt