import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

# VGG = torchvision.models.vgg19
from .base_network import vgg19
from .network_emb_study import RFB_modified
from .network_cs import Up


class Attention(nn.Module):
    def __init__(self, dim=768, heads=12, dropout=0.1):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        # self.qkv = nn.Linear(dim, dim * 3)
        self.q_func = nn.Linear(dim, dim)
        self.k_func = nn.Linear(dim, dim)
        self.v_func = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, x2=None):
        B, N, C = x.shape

        q = self.q_func(x2).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = self.k_func(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = self.v_func(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


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


if __name__ == '__main__':
    att = Attention()
    x = torch.rand((4, 768))
    x2 = att(x)
    import ipdb; ipdb.set_trace()

    test = UNet_SSL(3, 57)
    wtf = torch.zeros([1, 3, 224, 224], dtype=torch.float)
    wtf = test(wtf)
    import ipdb;    ipdb.set_trace()

