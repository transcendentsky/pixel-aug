"""

"""
from packaging import version
import numpy as np
# from tutils import tfilename
import os
import tensorboard as tb
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')  # Close plt.show() / Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import pandas as pd


def draw_heatmap_plot():
    # data = np.random.random((5,5))
    data = np.array([
                     [0.99,0.98,0.9,0.83,0.7],
                     [1,0.987,0.92,0.8,0.7],
                     [1,0.98,0.9,0.81,0.71],
                     [1,0.989,0.94,0.81,0.72],
                     [1,0.98,0.9,0.8,0.7],])
    ax = sns.heatmap(data)
    ax.set_ylabel("areas")
    ax.set_yticklabels([1.0, 0.8, 0.6, 0.4, 0.2])
    ax.set_xticklabels([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("high entropy areas%")
    plt.show()


def draw_line_plot():
    """ debug """
    fig = plt.figure()
    # fig.xlabel("Mutual info")
    # fig.ylabel("Performance (Mre)")
    ax = sns.lineplot(x=[3,2,1], y=[2.1,2,2.2], label="probmap")
    ax.set_title("fake data")
    ax.set_xlabel("Mutual information (MI)")
    fig.savefig("probmap_mi.png")
    plt.close()


def data_shift(d, move):
    new_d = np.zeros_like(d)
    for i, di in enumerate(d):
        mi = max(0, i + move)
        mi = min(mi, len(new_d)-1)
        new_d[i] = d[mi]
    print(new_d.shape)
    return new_d


def add_data_shift(data, index, d, move):
    for i, di in enumerate(d):
        mi = max(0, i + move)
        mi = min(mi, len(d)-1)
        data.loc[len(data.index)] = [index[i], d[mi]]


def read_data():
    pth = "D:\Documents/mi-noise/run-ssl_probmap_collect_sim_ps64_tb_train_gtv_0-tag-train.csv"
    data = pd.read_csv(pth)
    # print(data)
    fig = plt.figure()
    print(data.keys())
    d = data['Value'].values
    index = data['Step'].values
    data.drop(labels=["Wall time"], axis=1, inplace=True)
    # data2 = pd.DataFrame()
    for i in range(-30, 30, 4):
        add_data_shift(data, index, d, i)
    print("Smooth ok")
    sns.lineplot(x="Step", y="Value", data=data)
    # sns.lineplot(x=[1,2,3,4,5], y=[[1,2,3,4,5],[1,2,2,2,5],[1,4,4,4,5]])
    # plt.savefig("")
    # plt.imshow()
    plt.show()
    # import ipdb; ipdb.set_trace()



def read_all_data():
    preclip = 100
    dir_list = ['ssl_probmap', 'ssl', 'ssl_probmap3', "ssl_probmap0", "ssl_probmap0", "ssl_probmap0"]
    tag_list = ['collect_sim_ps64', 'ps64', 'debug', 'frac0.75', 'frac0.5', 'frac0.25']
    for dir_tag, tag in zip(dir_list, tag_list):
        ylist = []
        for i in range(3):
            pth = f"D:\Documents/mi-noise/run-{dir_tag}_{tag}_tb_train_gtv_{i}-tag-train.csv"
            data = pd.read_csv(pth)
            y = data['Value'].values[preclip:]
            ylist.append(y)
        xlist = data['Step'].values[preclip:]
        ylist = np.array(ylist)
        ylist = ylist.mean(axis=0)
        assert len(ylist) > 100, f"Got {ylist.shape} {tag}"
        sns.lineplot(x=xlist, y=ylist, label=tag)
    plt.show()




if __name__ == '__main__':
    # draw_line_plot()
    # read_data()
    # read_all_data()
    draw_heatmap_plot()