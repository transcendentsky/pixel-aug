import os.path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

import matplotlib.pyplot as plt


def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val


def draw_plt(val, val_name):
    """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    plt.figure()
    plt.plot([i.step for i in val], [j.value for j in val], label=val_name)
    """横坐标是step，迭代次数
    纵坐标是变量值"""
    plt.xlabel('step')
    plt.ylabel(val_name)
    plt.show()


def debug():
    tensorboard_path = 'D:\Documents\code\landmark/runs\ssl\ssl_probmap7\edge_inv/tb/train\gtv_0\events.out.tfevents.1661783657.miracle-103'
    val_name = 'train'
    val = read_tensorboard_data(tensorboard_path, val_name)
    draw_plt(val, val_name)


def test_prob7():
    dirname = 'D:\Documents\code\landmark/runs\ssl\ssl_probmap7'
    # experiments = ['baseline_ps64_3', "entr_ps64_inv2"]
    # experiments = ['baseline_ps64_3', 'edge1', 'entr_t0.3_(p6)', 'edge_inv', 'gaussian_ps64_pseudo', 'entr_ps64_inv2', 'mixprob', 'mixprob_1inv']
    experiments = ['baseline_ps64_3', 'entr_t0.3_(p6)', 'thres2_0', 'thres2_3', 'thres2_4.5']
    subdir = 'tb/train' # , 'entr_ps64_inv_0'
    lines = [f'gtv_{i}' for i in range(5)]
    plt.figure(figsize=(10,10))

    step = None
    for exp in experiments:
        res = None
        for line in lines:
            data = read_tensorboard_data(os.path.join(dirname, exp, subdir, line), 'train')
            values = np.array([j.value for j in data])[20:]
            res = values if res is None else res * values
            # step = [i.step for i in data] if step is None else step
        plt.plot(np.arange(0, len(res)), res, label=exp)
    plt.legend()
    plt.show()


def test_hand_probmap():
    dirname = 'D:\Documents\code\landmark/runs\ssl\ssl_hand_probmap'
    experiments = ['baseline', 't0.1']
    subdir = 'tb/train'
    lines = [f'gtv_{i}' for i in range(5)]
    plt.figure(figsize=(10,10))

    step = None
    for exp in experiments:
        res = None
        for line in lines:
            data = read_tensorboard_data(os.path.join(dirname, exp, subdir, line), 'train')
            values = np.array([j.value for j in data])[20:200]
            res = values if res is None else res * values
            # step = [i.step for i in data] if step is None else step
        plt.plot(np.arange(0, len(res)), res, label=exp)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    test_prob7()
    # test_hand_probmap()



