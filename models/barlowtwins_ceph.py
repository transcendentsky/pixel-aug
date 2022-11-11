from collections import OrderedDict
import os
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
# from .UNet3D import UNet3D, UNet3DEncoder_tmp
from models.network_emb_study import UNet_Pretrained
from tutils.trainer import LearnerModule


class BarlowProjector(nn.Module):
    def __init__(self, in_channel=16):
        super(BarlowProjector, self).__init__()
        # self.projector = nn.Sequential(nn.Conv3d(64, 512, kernel_size=1, padding=0),
        #                                nn.BatchNorm3d(1024),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv3d(1024, 2048, kernel_size=1, padding=0),
        #                                nn.BatchNorm3d(2048),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv3d(2048, 4096, kernel_size=1, padding=0)
        #                                )
        self.projector = nn.Sequential(nn.Linear(in_channel, 1024),
                                       nn.BatchNorm1d(1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 2048),
                                       nn.BatchNorm1d(2048),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(2048, 4096)
                                       )

    def forward(self, x):
        return self.projector(x)


class MyBarlowTwins(nn.Module):
    def __init__(self, config):
        super(MyBarlowTwins, self).__init__()
        self.config = config
        self.backbone = UNet_Pretrained(3, emb_len=config['training']['emb_len'])

    def forward(self, y1, y2):
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        return z1, z2




def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def exclude_bias_and_norm(p):
    return p.ndim == 1


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


# class BarlowTwins(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.backbone = torchvision.models.resnet50(zero_init_residual=True)
#         self.backbone.fc = nn.Identity()
#
#         # projector
#         sizes = [2048] + list(map(int, args.projector.split('-')))
#         layers = []
#         for i in range(len(sizes) - 2):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
#             layers.append(nn.BatchNorm1d(sizes[i + 1]))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
#         self.projector = nn.Sequential(*layers)
#
#         # normalization layer for the representations z1 and z2
#         self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
#
#     def forward(self, y1, y2):
#         z1 = self.projector(self.backbone(y1))
#         z2 = self.projector(self.backbone(y2))
#
#         # empirical cross-correlation matrix
#         c = self.bn(z1).T @ self.bn(z2)
#
#         # sum the cross-correlation matrix between all gpus
#         c.div_(self.args.batch_size)
#         torch.distributed.all_reduce(c)
#
#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#         off_diag = off_diagonal(c).pow_(2).sum()
#         loss = on_diag + self.args.lambd * off_diag
#         return loss