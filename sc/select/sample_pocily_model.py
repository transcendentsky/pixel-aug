"""
    copied from sample_policy2.py
    change to model

    # Greedy policy
"""
from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



def _fitness(x, matrix):
    v = matrix[x, :]
    vmax = v.max(axis=0)
    # assert len(vmax) == matrix.shape[1]
    return vmax.mean()


class FitModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FitModel, self).__init__()
        self.das = 0



if __name__ == '__main__':
    fitmodel = FitModel()
    import ipdb; ipdb.set_trace()