import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNETR as TR
# from monai.ne


class UNETR(nn.Module):
    def __init__(self, in_channels, n_classes, img_size=[384, 384], regression=True, *args, **kwargs):
        super(UNETR, self).__init__()
        self.regression = regression
        self.n_classes = n_classes
        final_channel = n_classes * 3 if regression else n_classes
        self.backbone = TR(in_channels=in_channels, out_channels=final_channel, spatial_dims=2, img_size=img_size, *args, **kwargs)

    def forward(self, x):
        x = self.backbone(x)

        if self.regression:
            heatmap = F.sigmoid(x[:, :self.n_classes, :, :])
            regression_x = x[:, self.n_classes:2 * self.n_classes, :, :]
            regression_y = x[:, 2 * self.n_classes:, :, :]
            return heatmap, regression_y, regression_x
        else:
            heatmap = F.sigmoid(x)
            return heatmap


class Swin_UNETR(nn.Module):
    def __init__(self, in_channels, n_classes, img_size=[384, 384], regression=True, *args, **kwargs):
        super(Swin_UNETR, self).__init__()
        self.regression = regression
        self.n_classes = n_classes
        final_channel = n_classes * 3 if regression else n_classes
        self.backbone = TR(in_channels=in_channels, out_channels=final_channel, spatial_dims=2, img_size=img_size, *args, **kwargs)

    def forward(self, x):
        x = self.backbone(x)

        if self.regression:
            heatmap = F.sigmoid(x[:, :self.n_classes, :, :])
            regression_x = x[:, self.n_classes:2 * self.n_classes, :, :]
            regression_y = x[:, 2 * self.n_classes:, :, :]
            return heatmap, regression_y, regression_x
        else:
            heatmap = F.sigmoid(x)
            return heatmap
