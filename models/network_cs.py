import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

# VGG = torchvision.models.vgg19
from .base_network import vgg19
from .network_emb_study import RFB_modified


class UNet_SSL(nn.Module):
    def __init__(self, n_channels, emb_len=32, non_local=False):
        super(UNet_SSL, self).__init__()
        self.n_channels = n_channels
        bilinear = True
        length_embedding = emb_len
        print("DEBUG setting: emb_len = ", emb_len)

        # self.vgg =  VGG(pretrained=True)
        self.vgg = vgg19(pretrained=True)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)
        self.trans_5 = nn.Conv2d(512, length_embedding, kernel_size=1, padding=0)
        self.trans_4 = nn.Conv2d(256, length_embedding, kernel_size=1, padding=0)
        self.trans_3 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        self.trans_2 = nn.Conv2d(64, length_embedding, kernel_size=1, padding=0)
        self.trans_1 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        self.trans_0 = nn.Conv2d(64, length_embedding, kernel_size=1, padding=0)

        # Kernel builder
        emb_len_last_layer = 64
        self.build_kernel_1 = nn.Sequential(nn.Linear(emb_len_last_layer, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, emb_len_last_layer),
                                            )
        self.build_kernel_2 = nn.Sequential(nn.Linear(emb_len_last_layer, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, emb_len_last_layer),
                                            )
        self.build_kernel_3 = nn.Sequential(nn.Linear(emb_len_last_layer, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(128, emb_len_last_layer),
                                            )

        if non_local:
            print("DEBUG setting: use non_local: ", non_local)
            self.non_local_5 = RFB_modified(512, 512)
            self.non_local_4 = RFB_modified(256, 256)
            self.non_local_3 = RFB_modified(128, 128)
        self.non_local = non_local

        # self.test_tensor = torch.Tensor([0., 1., 3.])

    def forward(self, x, landmark_feature=None):
        _, features = self.vgg(x, get_features=True)

        x = self.up1(features[4], features[3])
        if self.non_local:
            fea_5 = self.trans_5(self.non_local_5(features[4]))
            fea_4 = self.trans_4(self.non_local_4(x))
        else:
            fea_5 = self.trans_5(features[4])
            fea_4 = self.trans_4(x)
        x = self.up2(x, features[2])
        if self.non_local:
            fea_3 = self.trans_3(self.non_local_3(x))
        else:
            fea_3 = self.trans_3(x)
        x = self.up3(x, features[1])
        fea_2 = self.trans_2(x)
        x = self.up4(x, features[0])
        fea_1 = self.trans_1(x)
        x = self.up5(x)
        fea_0 = self.trans_0(x)

        # print("landmark_shape: ", landmark_feature.shape)
        # import ipdb; ipdb.set_trace()
        kernel_1 = self.build_kernel_1(landmark_feature)
        kernel_2 = self.build_kernel_2(landmark_feature)
        kernel_3 = self.build_kernel_3(landmark_feature)
        # print("kernel_shape: ", kernel_1.shape, kernel_2.shape, kernel_3.shape)

        heatmap      = F.conv1d(input=x, weight=kernel_1.unsqueeze(-1).unsqueeze(-1), bias=None, stride=1, padding=0, dilation=1, groups=1)
        heatmap = F.sigmoid(heatmap)
        regression_y = F.conv1d(input=x, weight=kernel_2.unsqueeze(-1).unsqueeze(-1), bias=None, stride=1, padding=0, dilation=1, groups=1)
        regression_x = F.conv1d(input=x, weight=kernel_3.unsqueeze(-1).unsqueeze(-1), bias=None, stride=1, padding=0, dilation=1, groups=1)
        # print("debug: heatmap.shape", heatmap.shape)
        return [fea_5, fea_4, fea_3, fea_2, fea_1, fea_0, heatmap, regression_y, regression_x]


class UNet_Pretrained(nn.Module):
    def __init__(self, n_channels, emb_len=32, non_local=False):
        super(UNet_Pretrained, self).__init__()
        self.n_channels = n_channels
        bilinear = True
        length_embedding = emb_len
        print("DEBUG setting: emb_len = ", emb_len)

        # self.vgg =  VGG(pretrained=True)
        self.vgg = vgg19(pretrained=True)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)
        self.trans_5 = nn.Conv2d(512, length_embedding, kernel_size=1, padding=0)
        self.trans_4 = nn.Conv2d(256, length_embedding, kernel_size=1, padding=0)
        self.trans_3 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        self.trans_2 = nn.Conv2d(64, length_embedding, kernel_size=1, padding=0)
        self.trans_1 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        self.trans_0 = nn.Conv2d(64, length_embedding, kernel_size=1, padding=0)

        if non_local:
            print("DEBUG setting: use non_local: ", non_local)
            self.non_local_5 = RFB_modified(512, 512)
            self.non_local_4 = RFB_modified(256, 256)
            self.non_local_3 = RFB_modified(128, 128)
        self.non_local = non_local

        # self.test_tensor = torch.Tensor([0., 1., 3.])

    def forward(self, x, mlp=False, ret_last_layer=False):
        # _, features = self.vgg.features(x, get_features=True)
        # # For torchvsion of later version
        # features = self.vgg.features(x)
        _, features = self.vgg(x, get_features=True)

        x = self.up1(features[4], features[3])
        if self.non_local:
            fea_5 = self.trans_5(self.non_local_5(features[4]))
            fea_4 = self.trans_4(self.non_local_4(x))
        else:
            fea_5 = self.trans_5(features[4])
            fea_4 = self.trans_4(x)
        x = self.up2(x, features[2])
        if self.non_local:
            fea_3 = self.trans_3(self.non_local_3(x))
        else:
            fea_3 = self.trans_3(x)
        x = self.up3(x, features[1])
        fea_2 = self.trans_2(x)
        x = self.up4(x, features[0])
        fea_1 = self.trans_1(x)
        x = self.up5(x)
        fea_0 = self.trans_0(x)
        if ret_last_layer:
            return [fea_5, fea_4, fea_3, fea_2, fea_1, fea_0, x]

        return [fea_5, fea_4, fea_3, fea_2, fea_1, fea_0]


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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


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


class OutConv_Sigmoid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_Sigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    test = UNet_Pretrained(3, 57)
    wtf = torch.zeros([1, 3, 224, 224], dtype=torch.float)
    wtf = test(wtf)
    import ipdb;

    ipdb.set_trace()

