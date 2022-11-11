import torch
from typing import Union, List, Dict, Any, cast
from torch import nn
import torch.nn.functional as F

# from torchvision.models.vgg import VGG

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(**kwargs)
    # model = MyVgg(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

cfgs2 = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']]

class VGG_CS1(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
            init_weights: bool = True
    ) -> None:
        super(VGG_CS1, self).__init__()
        batch_norm = False
        self.features0 = make_layers(cfgs2[0], batch_norm=batch_norm, in_channels=3) # 64
        self.features1 = make_layers(cfgs2[1], batch_norm=batch_norm, in_channels=64)
        self.features2 = make_layers(cfgs2[2], batch_norm=batch_norm, in_channels=128)
        self.features3 = make_layers(cfgs2[3], batch_norm=batch_norm, in_channels=256)
        self.features4 = make_layers(cfgs2[4], batch_norm=batch_norm, in_channels=512)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, _x: torch.Tensor):
        """
        SENet style
        """
        # f0, f1, f2, f3, f4 = feas
        # assert len(f0.shape) == 4, f"Got f0.shape {f0.shape}"

        # import ipdb; ipdb.set_trace()
        x0 = self.features0(_x)
        # import ipdb; ipdb.set_trace()

        x1 = self.features1(x0)

        x2 = self.features2(x1)

        x3 = self.features3(x2)

        x4 = self.features4(x3)

        # x = self.avgpool(x4)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return [x0, x1, x2, x3, x4]

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class VGG_CS2(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
            init_weights: bool = True
    ) -> None:
        super(VGG_CS2, self).__init__()
        batch_norm = False
        self.features0 = make_layers(cfgs2[0], batch_norm=batch_norm, in_channels=3) # 64
        self.features1 = make_layers(cfgs2[1], batch_norm=batch_norm, in_channels=64)
        self.features2 = make_layers(cfgs2[2], batch_norm=batch_norm, in_channels=128)
        self.features3 = make_layers(cfgs2[3], batch_norm=batch_norm, in_channels=256)
        self.features4 = make_layers(cfgs2[4], batch_norm=batch_norm, in_channels=512)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

        # length_embedding = 32
        self.trans_0 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.trans_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.trans_2 = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.trans_3 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.trans_4 = nn.Conv2d(512, 512, kernel_size=1, padding=0)

    def forward(self, _x: torch.Tensor, feas):
        """
        SENet style
        """
        f0, f1, f2, f3, f4 = feas
        if len(f0.shape) == 3:
            f0 = f0.unsqueeze(0)
            f1 = f1.unsqueeze(0)
            f2 = f2.unsqueeze(0)
            f3 = f3.unsqueeze(0)
            f4 = f4.unsqueeze(0)
        assert len(f0.shape) == 4, f"Got f0.shape {f0.shape}"

        x0 = self.features0(_x)
        f0 = self.trans_0(f0)
        # import ipdb; ipdb.set_trace()
        assert x0.shape[1] == f0.shape[1], f"Got {x0.shape, f0.shape}"
        # x0 = F.conv2d(x0, f0, stride=1)
        x0 = x0 * f0

        x1 = self.features1(x0)
        f1 = self.trans_1(f1)
        x1 = x1 * f1
        # import ipdb; ipdb.set_trace()

        x2 = self.features2(x1)
        f2 = self.trans_2(f2)
        x2 = x2 * f2
        # import ipdb; ipdb.set_trace()

        x3 = self.features3(x2)
        f3 = self.trans_3(f3)
        x3 = x3 * f3
        # import ipdb; ipdb.set_trace()

        x4 = self.features4(x3)
        f4 = self.trans_4(f4)
        x4 = x4 * f4
        # import ipdb; ipdb.set_trace()

        # x = self.avgpool(x4)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return [x0, x1, x2, x3, x4]

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # vgg19
}


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels=3) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)