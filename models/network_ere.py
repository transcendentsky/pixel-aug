import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# VGG = torchvision.models.vgg19
from .base_network import vgg19


def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


def nll_across_batch(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=(2, 3)))


class UNet_Pretrained(nn.Module):
    def __init__(self, n_channels=3, n_classes=19, regression=False):
        # set "use_sigmoid = False" for BCELossWithLogits
        # n_channels=3, n_classes=19
        super(UNet_Pretrained, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg = vgg19(pretrained=True)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        # self.up5 = Up(128, 64, bilinear) # original code
        self.up5 = Up(128, 128, bilinear) # align to ssl network

        self.regression = regression
        # self.use_sigmoid = use_sigmoid
        final_channel = self.n_classes * 3 if regression else self.n_classes
        # self.final = nn.Conv2d(64, final_channel, kernel_size=1, padding=0) # original code
        self.final = nn.Conv2d(128, final_channel, kernel_size=1, padding=0) # align to ssl network
        self.temperatures = nn.Parameter(torch.ones(1, n_classes, 1, 1), requires_grad=False)


    def forward(self, x, no_sigmoid=False):
        _, features = self.vgg(x, get_features=True)

        x = self.up1(features[4], features[3])
        x = self.up2(x, features[2])
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        x = self.up5(x)

        x = self.final(x)

        if no_sigmoid:
            return x

        if self.regression:
            heatmap = F.sigmoid(x[:, :self.n_classes, :, :])
            regression_x = x[:, self.n_classes:2 * self.n_classes, :, :]
            regression_y = x[:, 2 * self.n_classes:, :, :]
            return heatmap, regression_y, regression_x
        else:
            heatmap = F.sigmoid(x)
            return heatmap

    def scale(self, x):
        y = x / self.temperatures
        return y

    def forward_no_sigmoid(self, x):
        _, features = self.vgg(x, get_features=True)

        x = self.up1(features[4], features[3])
        x = self.up2(x, features[2])
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        x = self.up5(x)

        x = self.final(x)
        heatmap = x
        return heatmap


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)