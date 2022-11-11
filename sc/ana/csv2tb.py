import pandas as pd
import numpy as np
from tutils import tfilename
import os
import numpy as np
from tutils import trans_init, trans_args, dump_yaml, tfilename, CSVLogger
import argparse
import seaborn as sns

import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
# plt.style.use('ggplot')



def read_data(path=None):
    data = pd.read_csv(path)
    # print(data)
    # import ipdb; ipdb.set_trace()
    return data


def draw(lines):
    for line in lines:
        x = line[0]
        y = line[1]
        fig = sns.lineplot(x=x, y=y, label="GU$\mathregular{^2}$Net")


def _draw_debug(ori_data):
    data = ori_data[1:]
    # fig = sns.lineplot(x='mre', y='sim_layer_point_4', data=data)
    # fig = sns.lineplot(x='mre', y='sim_layer_point_3', data=data)
    # fig = sns.lineplot(x='mre', y='sim_layer_point_2', data=data)
    # fig = sns.lineplot(x='mre', y='sim_layer_point_1', data=data)
    # fig = sns.lineplot(x='mre', y='sim_layer_point_0', data=data)

    data['sim_gap'] = data['max_sim'] - data['lm_sim']
    labels = []
    # labels += [f'sim_layer_point_{i}' for i in range(5)]
    # labels += [f'sim_layer_max_{i}' for i in range(5)]
    labels += ['max_sim', 'lm_sim', 'sim_gap']
    for label in labels:
        fig = sns.lineplot(x='mre', y=label, data=data, label=label)

    # fig.legend()
    scatter_fig = fig.get_figure()
    scatter_fig.savefig("debug_csv2tb"+".png", dpi=400)
    # fig.imshow()

def _draw_debug2():
    # path = "D:\Documents\git-clone\landmark/runs\ssl\interpolate\collect_sim/best_record/record.csv"
    # path2 = "D:\Documents\git-clone\landmark/runs\ssl\ssl_mi\collect_sim/best_record/record.csv"
    # path3 = "D:\Documents\git-clone\landmark/runs\ssl\ssl_probmap\collect_sim/best_record/record.csv"
    path = "D:\Documents\git-clone\landmark/runs\ssl\interpolate\collect_sim/all_record/record.csv"
    path2 = "D:\Documents\git-clone\landmark/runs\ssl\ssl_mi\collect_sim/all_record/record.csv"
    path3 = "D:\Documents\git-clone\landmark/runs\ssl\ssl_probmap\collect_sim/all_record/record.csv"
    path4 = "D:\Documents\git-clone\landmark/runs\ssl\emd\debug/best_record/record.csv"

    th = 3
    data1 = read_data(path)[th:]
    data2 = read_data(path2)[th:]
    data3 = read_data(path3)[th:]
    data4 = read_data(path4)[-6:]

    # print(data1.dtypes)

    for data, name in zip([data1, data2, data3, data4], ['ip', 'mi', 'prob', 'emd']):
    # for data, name in zip([data1, data2, data3], ['ip', 'mi', 'prob']):
        labels = []
        # labels += ['max_sim', 'lm_sim', 'sim_gap']
        # labels = ['sim_gap']
        # labels = ['lm_sim']
        data['sim_gap'] = data['max_sim'] - data['lm_sim']
        data['id'] = range(len(data))
        for i in range(5):
            data[f'sim_l_gap_{i}'] = data[f'sim_layer_max_{i}'] - data[f'sim_layer_point_{i}']
        data[f'sim_l_gap_sum'] = data[f'sim_l_gap_0'] + data[f'sim_l_gap_1'] + data[f'sim_l_gap_2'] + data[f'sim_l_gap_3'] + data[f'sim_l_gap_4']
        data[f'sim_layer_point_sum'] = data[f'sim_layer_point_{0}'] + data[f'sim_layer_point_{1}'] + data[
            f'sim_layer_point_{2}'] + data[f'sim_layer_point_{3}'] + data[f'sim_layer_point_{4}']
        data[f'sim_layer_max_sum'] = data[f'sim_layer_max_{0}'] + data[f'sim_layer_max_{1}'] + data[
            f'sim_layer_max_{2}'] + data[f'sim_layer_max_{3}'] + data[f'sim_layer_max_{4}']

        labels += ['max_sim']
        # labels += [f'sim_l_gap_{i}' for i in range(5)]
        # labels += [f'sim_l_gap_sum']
        # labels += [f'sim_l_gap_0']
        # labels += ['mre']
        # labels += [f'sim_layer_point_sum']
        # labels += [f'sim_layer_max_sum']
        for label in labels:
            fig = sns.lineplot(x=label, y='mre', data=data, label=name + "_" + label)
            # fig = sns.lineplot(x='mre', y=label, data=data, label=name + "_" + label)
            # fig = sns.lineplot(x='id', y=label, data=data, label=name + "_" + label)

    scatter_fig = fig.get_figure()
    scatter_fig.savefig("debug_csv2tb"+".png", dpi=400)


def _draw_debug3():
    path = 'D:\Documents\git-clone\landmark/runs/ana\max_sim\collect_sim/record_train_near.csv'
    path2 = 'D:\Documents\git-clone\landmark/runs/ana\max_sim\collect_sim/record_test_near.csv'
    path3 = 'D:\Documents\git-clone\landmark/runs/ana\max_sim\collect_sim/record_test_linear.csv'

    data1 = read_data(path)
    data2 = read_data(path2)
    data3 = read_data(path3)

    for data, name in zip([data1, data2, data3], ['train-near', 'test-near', 'test-linear']):
        # data.drop(tag='ip_debug')
        # data.drop(tag='ip_debug4')
        data = data.drop(index= [2, 8])
        print(data)
        # import ipdb; ipdb.set_trace()
        data['sim_gap'] = data['max_sim'] - data['lm_sim']
        data['id'] = range(len(data))
        for i in range(5):
            data[f'sim_l_gap_{i}'] = data[f'sim_layer_max_{i}'] - data[f'sim_layer_point_{i}']
        data[f'sim_l_gap_sum'] = data[f'sim_l_gap_0'] + data[f'sim_l_gap_1'] + data[f'sim_l_gap_2'] + data[f'sim_l_gap_3'] + data[f'sim_l_gap_4']
        data[f'sim_layer_point_sum'] = data[f'sim_layer_point_{0}'] + data[f'sim_layer_point_{1}'] + data[
            f'sim_layer_point_{2}'] + data[f'sim_layer_point_{3}'] + data[f'sim_layer_point_{4}']
        data[f'sim_layer_max_sum'] = data[f'sim_layer_max_{0}'] + data[f'sim_layer_max_{1}'] + data[
            f'sim_layer_max_{2}'] + data[f'sim_layer_max_{3}'] + data[f'sim_layer_max_{4}']

        labels = []
        # labels += ['max_sim']
        labels += ['lm_sim']
        # labels += [f'sim_l_gap_{i}' for i in range(5)]
        # labels += [f'sim_l_gap_sum']
        # labels += [f'sim_l_gap_4']
        # labels += ['mre']
        # labels += [f'sim_layer_point_sum']
        # labels += [f'sim_layer_max_sum']
        for label in labels:
            fig = sns.lineplot(x=label, y='mre', data=data, label= name + "_" + label)
            # fig = sns.lineplot(x='mre', y=label, data=data, label=name + "_" + label)
            # fig = sns.lineplot(x='id', y=label, data=data, label=name + "_" + label)

    scatter_fig = fig.get_figure()
    scatter_fig.savefig("debug2_csv2tb"+".png", dpi=400)


def analysis():
    # path = "/home1/quanquan/code/landmark/code/runs/ssl/interpolate/collect_sim/best_record/record.csv"
    path  = "D:\Documents\git-clone\landmark/runs\ssl\interpolate\collect_sim/best_record/record.csv"
    path2 = "D:\Documents\git-clone\landmark\runs\ssl\ssl_mi\collect_sim\best_record\record.csv"
    path3 = "D:\Documents\git-clone\landmark\runs\ssl\ssl_probmap\collect_sim\best_record\record.csv"
    # data = read_data(path)
    # _draw_debug(data)

    # line1 = [data['mre'], data['']]
    # lines = [line1]


if __name__ == '__main__':
    # from tutils import trans_init, trans_args
    # ex_config = {"base": {"base_dir": "../runs/ana/"}}
    # trans_init(file=__file__, ex_config=ex_config)

    # analysis()
    _draw_debug3()