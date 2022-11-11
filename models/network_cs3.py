import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

# VGG = torchvision.models.vgg19
from .base_network import vgg19
from .network_emb_study import RFB_modified
from .network_cs import Up
from .vgg2 import VGG_CS1, VGG_CS2


class UNet_1(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, regression=True):
        # n_channels=3, n_classes=19
        assert n_channels == 3, f"Got n_channels = {n_channels}"
        super(UNet_1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        print("UNet Model: Regression = True")

        self.vgg = VGG_CS1()
        self.vgg2 = VGG_CS2()
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)

        self.regression = regression
        final_channel = self.n_classes * 3 if regression else self.n_classes
        self.final = nn.Conv2d(64, final_channel, kernel_size=1, padding=0)

    def get_vectors_from_feature_maps(self, point, fea):
        img_size = 384
        scale = img_size // fea.shape[-1]
        f0 = torch.stack([fea[[id], :, point[id][0] // scale, point[id][1] // scale] \
                          for id in range(point.shape[0])]).squeeze()
        return f0

    def forward(self, img, img2, point):
        # Label_Encoder
        x0, x1, x2, x3, x4 = self.vgg(img2)
        # import ipdb; ipdb.set_trace()
        f0 = self.get_vectors_from_feature_maps(point, x0).unsqueeze(-1).unsqueeze(-1)
        f1 = self.get_vectors_from_feature_maps(point, x1).unsqueeze(-1).unsqueeze(-1)
        f2 = self.get_vectors_from_feature_maps(point, x2).unsqueeze(-1).unsqueeze(-1)
        f3 = self.get_vectors_from_feature_maps(point, x3).unsqueeze(-1).unsqueeze(-1)
        f4 = self.get_vectors_from_feature_maps(point, x4).unsqueeze(-1).unsqueeze(-1)
        # import ipdb; ipdb.set_trace()

        # Image_Encoder
        x0, x1, x2, x3, x4 = self.vgg2(img, (f0, f1, f2, f3, f4))

        # Shared Header
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.up5(x)
        x = self.final(x)

        if self.regression:
            heatmap = F.sigmoid(x[:, :self.n_classes, :, :])
            regression_x = x[:, self.n_classes:2 * self.n_classes, :, :]
            regression_y = x[:, 2 * self.n_classes:, :, :]
            return heatmap, regression_y, regression_x
        else:
            heatmap = F.sigmoid(x)
            return heatmap

    # def infer(self, img, crops, points):
    #     x0, x1, x2, x3, x4 = self.vgg2(img, (f0, f1, f2, f3, f4))



class UNet_2(nn.Module):
    def __init__(self, n_channels, n_classes, regression=False):
        # n_channels=3, n_classes=19
        super(UNet_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg = vgg19(pretrained=True)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)

        self.regression = regression
        final_channel = self.n_classes * 3 if regression else self.n_classes
        self.final = nn.Conv2d(64, final_channel, kernel_size=1, padding=0)

    def forward(self, x):
        x0, x1, x2, x3, x4, x = self.vgg(x, get_features=True)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.up5(x)

        x = self.final(x)

        if self.regression:
            heatmap = F.sigmoid(x[:, :self.n_classes, :, :])
            regression_x = x[:, self.n_classes:2 * self.n_classes, :, :]
            regression_y = x[:, 2 * self.n_classes:, :, :]
            return heatmap, regression_y, regression_x
        else:
            heatmap = F.sigmoid(x)
            return heatmap

if __name__ == '__main__':
    import numpy as np
    model = UNet_1()
    x1 = torch.ones((2, 3, 384, 384))
    x2 = torch.ones((2, 3, 192, 192))
    lm = np.array([[5,5], [8,8]])
    heatmap, regression_y, regression_x = model(x1, x2, lm)


    import ipdb;    ipdb.set_trace()

