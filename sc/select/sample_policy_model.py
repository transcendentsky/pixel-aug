"""
    copied from sample_policy2.py
    change to model

    # Greedy policy
    LoFTR Landmark discovery + SIFT
    LoFTR matching

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
    from models.pointer_net import Encoder, Attention, Decoder, PointerNet
    matrix = np.load("data_max_list_all.npy")
    import ipdb; ipdb.set_trace()
    # fitmodel = FitModel()
    a = 0.1
    x = f"{{({a})}}"
    print(x)
    # encoder = Encoder(embedding_dim=128,
    #              hidden_dim=512,
    #              n_layers=2,
    #              dropout=0.,
    #              bidir=False)
    net = PointerNet(input_dim=2,
        embedding_dim=128,
                 hidden_dim=512,
                 lstm_layers=2,
                 dropout=0.,
                 bidir=False)
    data = torch.ones((4, 5, 2))
    target = torch.ones((4, 5))
    output, pointers = net(data)
    print(output.shape)
    import ipdb; ipdb.set_trace()

    # for _ in range(10):
    #     pointer = net(matrix)
    #     fit = _fitness(pointer, matrix)
    #     loss = fit