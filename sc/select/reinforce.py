# coding: utf-8
"""
    “Planning ” ：“规划问题”，在完全了解 MDP 结构的情况下，求解最优的 Policy。
"""

import numpy as np


def _fitness(x, matrix):
    v = matrix[x, :]
    vmax = v.max(axis=0)
    # assert len(vmax) == matrix.shape[1]
    return vmax.mean()


if __name__ == '__main__':

    matrix = np.load("data_max_list_all.npy")