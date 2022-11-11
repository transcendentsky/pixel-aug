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


def _fitness(x, matrix):
    v = matrix[x, :]
    vmax = v.max(axis=0)
    # assert len(vmax) == matrix.shape[1]
    return vmax.mean()


def intergate_datalist():
    data_list = []
    for idx in range(150):
        # "/home1/quanquan/code/landmark/code/runs-ana/sift1/max_list/data_max_list_oneshot_0.npy"
        print(f"****  idx: {idx} ", end=" \r")
        # This is the
        data = np.load(tfilename(
            f'/home1/quanquan/code/landmark/code/runs-ana/' + 'sift2' + f'/max_list/data_max_list_oneshot_{idx}.npy'))
        data = data[:, :, -1]
        data_list.append(data)
        # import ipdb; ipdb.set_trace()
    data_np = np.array(data_list)
    np.save(f"data_max_list_all.npy", data_np)
    return data_np


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
    plt.xticks(x_pos[:-1], labels[:-1], fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig(fname)
    plt.close()
    print("Drawed img: ", fname)


def draw_bar_sim_n1():
    mre_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n1_1/mre_n1_list.npy"
    mre = np.load(mre_list)
    print(mre.shape)
    # print(mre)
    thresholds = np.linspace(2.9, 4.5, num=17)
    thresholds[-1] = 999
    print(thresholds, thresholds.shape)
    length_collect = []
    pre_len = 0
    for i in range(len(thresholds)-1):
        ind = np.where(mre<=thresholds[i+1] )[0]
        # print(ind)
        # if len(ind) <= 0:
        #     continue
        length_collect.append(len(ind) - pre_len)
        pre_len = len(ind)
        # thresholds.append(f"{i*0.05:.2f}")
        print("mmm: ", len(ind), length_collect)
        # import ipdb; ipdb.set_trace()
    # thresholds.append(f"{(end+1)*0.05:.2f}")
    length_collect = np.array(length_collect) / 150
    draw_bar(thresholds, length_collect, fname=f"tbar_mre_n1.png", color="blue", xlabel="Mean Radial Error (MRE)", ylabel="Quantity (%)")
    draw_bar(thresholds, length_collect, fname=f"tbar_mre_n1.pdf", color="blue", xlabel="Mean Radial Error (MRE)", ylabel="Quantity (%)")


def draw_bar_maxsim():
    from einops import rearrange
    n = 10
    maxsim_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{n}_1/maxconf_n{n}_list.npy"
    maxsim_list_sift = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{n}_sift/max_list_sift_n{n}.npy"
    maxsim_list = np.load(maxsim_list)
    maxsim_list_sift = np.load(maxsim_list_sift)
    ms = rearrange(maxsim_list, "x (y z) -> z x y", y=150, z=19)
    ms = np.mean(ms, axis=-1)
    # ms = rearrange(ms, "z x")
    lm_idx = 3
    ms0 = ms[lm_idx, :]
    maxsim_list_sift2 = np.mean(np.mean(maxsim_list_sift, axis=-1), axis=-1)
    # import ipdb; ipdb.set_trace()
    maxsim_point = rearrange(maxsim_list_sift, "x y z -> (x z) y")
    maxsim_point2 = np.mean(maxsim_point, axis=-1)
    maxsim_list2 = maxsim_list.mean(-1)
    # import ipdb; ipdb.set_trace()
    draw_scatter(maxsim_list2, maxsim_list_sift2, fname=f"maxsim_relation_n{n}.png", xlabel="key points of interest", ylabel="potential key points")

    # inds = np.arange(0, 150)
    # maxsim_list2 =
    cc = np.corrcoef(maxsim_list2, maxsim_list_sift2)
    print(f"n={n},  cc: ", cc)
    return
    # interval = 0.05
    # mm = maxsim_point2 / interval
    # mm = mm.astype(int)
    #
    # length_collect = []
    # thresholds = []
    # for i in range(20):
    #     ind = np.where(mm==np.array(i).astype(int))[0]
    #     length_collect.append(len(ind))
    #     thresholds.append(f"{i*0.05:.2f}")
    #     # thresholds.append(str(i*0.05))
    #     print(len(ind))
    #     # import ipdb; ipdb.set_trace()
    # length_collect = np.array(length_collect) / len(mm)
    # thresholds = thresholds[1:-3]
    # length_collect = length_collect[1:-3]
    # print(length_collect)
    # print(len(length_collect), len(thresholds))
    # draw_bar(thresholds, length_collect, fname="tbar.pdf" , xlabel="thresholds", ylabel="num")

    #
    interval = 0.05
    mm = ms0 / interval
    mm = mm.astype(int)
    length_collect = []
    thresholds = []
    for i in range(20):
        ind = np.where(mm==np.array(i).astype(int))[0]
        if len(ind) <= 0:
            continue
        end = i
        length_collect.append(len(ind))
        thresholds.append(f"{i*0.05:.2f}")
        print(len(ind))
        # import ipdb; ipdb.set_trace()
    thresholds.append(f"{(end+1)*0.05:.2f}")
    length_collect = np.array(length_collect) / 150

    # thresholds0 = thresholds[6:15]
    # length_collect0 = length_collect[6:14]
    thresholds0 = thresholds
    length_collect0 = length_collect
    print("length_collect.sum() ", length_collect0.sum())
    print("len, len: ", len(length_collect0), len(thresholds0))
    # manual selectio
    draw_bar(thresholds0, length_collect0, fname=f"tbar_lm{lm_idx}.png", color="blue", xlabel="similarity(thresholds)", ylabel="num(%)")
    draw_bar(thresholds0, length_collect0, fname=f"tbar_lm{lm_idx}.pdf", color="blue", xlabel="similarity(thresholds)", ylabel="num(%)")
    # draw_bar(thresholds, length_collect, fname=f"tbar_lm{lm_idx}_all.pdf", color="blue", xlabel="similarity(thresholds)", ylabel="num(%)")
    # draw_bar(thresholds, length_collect, fname=f"tbar_lm{lm_idx}_all.png", color="blue", xlabel="similarity(thresholds)", ylabel="num(%)")
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    # draw_bar_maxsim()
    draw_bar_sim_n1()

