import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import math
import warnings
from models.conformer_det_backbone import Conformer as _Conformer
from torchvision.utils import save_image


class ConformerDet(nn.Module):
    def __init__(self, config):
        super(ConformerDet, self).__init__()

        self.backbone = _Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                                    num_heads=6, mlp_ratio=4, qkv_bias=True)
        # length_embedding = 128
        # self.trans_5 = nn.Conv2d(512, length_embedding, kernel_size=1, padding=0)
        # self.trans_4 = nn.Conv2d(256, length_embedding, kernel_size=1, padding=0)
        # self.trans_3 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        # self.trans_2 = nn.Conv2d(64, length_embedding, kernel_size=1, padding=0)
        # self.trans_1 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        # self.trans_0 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        self.n_classes = config['dataset']['n_cls']
        self.im_size = config['dataset']['im_size']
        assert self.n_classes == 19

        # self.trans = nn.Conv2d(64, self.n_classes*3, kernel_size=1, padding=0)
        self.trans0 = nn.Conv2d(64, self.n_classes*3, kernel_size=1, padding=0)
        self.trans1 = nn.Conv2d(128, self.n_classes*3, kernel_size=1, padding=0)
        self.trans2 = nn.Conv2d(256, self.n_classes*3, kernel_size=1, padding=0)
        self.trans3 = nn.Conv2d(256, self.n_classes*3, kernel_size=1, padding=0)

    def forward(self, x, return_features=False):
        # print("[*] debug! ", x.shape)
        x = self.backbone(x)
        # print(len(x))
        # for i, xi in enumerate(list(x)):
        #     _x = torch.mean(xi, dim=1)
        #     save_image(_x, f'tmp/conformer_x{i}.png')
        #     print("debug ", xi.shape)

        x0, x1, x2, x3 = x[2], x[3], x[4], x[5]

        # print("[*] debug! ", x.shape)
        masks0 = self.trans0(x0)
        masks1 = self.trans1(x1)
        masks2 = self.trans2(x2)
        masks3 = self.trans3(x3)
        # print("[*] debug! ", masks.shape)
        H, W = self.im_size, self.im_size
        masks0 = F.interpolate(masks0, size=(H, W), mode="bilinear")
        masks1 = F.interpolate(masks1, size=(H, W), mode="bilinear")
        masks2 = F.interpolate(masks2, size=(H, W), mode="bilinear")
        masks3 = F.interpolate(masks3, size=(H, W), mode="bilinear")

        heatmap = []
        regression_x = []
        regression_y = []

        heatmap      += [F.sigmoid(masks0[:, :self.n_classes, :, :])]
        regression_x += [masks0[:, self.n_classes:2 * self.n_classes, :, :]]
        regression_y += [masks0[:, 2 * self.n_classes:, :, :]]

        heatmap      += [F.sigmoid(masks1[:, :self.n_classes, :, :])]
        regression_x += [masks1[:, self.n_classes:2 * self.n_classes, :, :]]
        regression_y += [masks1[:, 2 * self.n_classes:, :, :]]

        heatmap      += [F.sigmoid(masks2[:, :self.n_classes, :, :])]
        regression_x += [masks2[:, self.n_classes:2 * self.n_classes, :, :]]
        regression_y += [masks2[:, 2 * self.n_classes:, :, :]]

        heatmap      += [F.sigmoid(masks3[:, :self.n_classes, :, :])]
        regression_x += [masks3[:, self.n_classes:2 * self.n_classes, :, :]]
        regression_y += [masks3[:, 2 * self.n_classes:, :, :]]
        return heatmap, regression_x, regression_y
