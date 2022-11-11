# from __future__ import annotations
import torch
import torchvision
import numpy as np
from tutils import tfilename, trans_args, trans_init, print_dict
import torchvision.transforms as transforms
from torchvision.models import resnet50, wide_resnet50_2
from torch import nn as nn
from typing import Tuple
# import torch.nn.Tensor as Tensor

import random
from copy import deepcopy
from typing import Optional

import torch
from torch import nn, Tensor
from torch.distributions import Categorical

# from dda.operations import *
from dda.operations import ShearX, ShearY, TranslateY, TranslateY, Rotate, HorizontalFlip, Invert, Solarize, Posterize, Contrast, \
    Saturate, Brightness, Sharpness, AutoContrast, Equalize


Tensor = torch.Tensor


class Discriminator(nn.Module):
    def __init__(self,
                 base_module: nn.Module
                 ):
        super(Discriminator, self).__init__()
        self.base_model = base_module
        num_features = self.base_model.fc.in_features
        num_class = self.base_model.fc.out_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_class)
        self.discriminator = nn.Sequential(nn.Linear(num_features, num_features),
                                           nn.ReLU(),
                                           nn.Linear(num_features, 1))

    def forward(self,
                input: Tensor
                ) -> Tuple[Tensor, Tensor]:
        x = self.base_model(input)
        return self.classifier(x), self.discriminator(x).view(-1)





class SubPolicyStage(nn.Module):
    def __init__(self,
                 operations: nn.ModuleList,
                 temperature: float,
                 ):
        super(SubPolicyStage, self).__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature

    def forward(self,
                input: Tensor
                ) -> Tensor:
        if self.training:
            return (torch.stack([op(input) for op in self.operations]) * self.weights.view(-1, 1, 1, 1, 1)).sum(0)
        else:
            return self.operations[Categorical(self.weights).sample()](input)

    @property
    def weights(self
                ):
        return self._weights.div(self.temperature).softmax(0)


class SubPolicy(nn.Module):
    def __init__(self,
                 sub_policy_stage: SubPolicyStage,
                 operation_count: int,
                 ):
        super(SubPolicy, self).__init__()
        self.stages = nn.ModuleList([deepcopy(sub_policy_stage) for _ in range(operation_count)])

    def forward(self,
                input: Tensor
                ) -> Tensor:
        for stage in self.stages:
            input = stage(input)
        return input


class Policy(nn.Module):
    def __init__(self,
                 operations: nn.ModuleList,
                 num_sub_policies: int,
                 temperature: float = 0.05,
                 operation_count: int = 2,
                 num_chunks: int = 4,
                 mean: Optional[Tensor] = None,
                 std: Optional[Tensor] = None,
                 ):
        super(Policy, self).__init__()
        self.sub_policies = nn.ModuleList([SubPolicy(SubPolicyStage(operations, temperature), operation_count)
                                           for _ in range(num_sub_policies)])
        self.num_sub_policies = num_sub_policies
        self.temperature = temperature
        self.operation_count = operation_count
        self.num_chunks = num_chunks
        if mean is None:
            self._mean, self._std = None, None
        else:
            self.register_buffer('_mean', mean)
            self.register_buffer('_std', std)

        for p in self.parameters():
            nn.init.uniform_(p, 0, 1)

    def forward(self,
                input: Tensor
                ) -> Tensor:
        # [0, 1] -> [-1, 1]

        if self.num_chunks > 1:
            out = [self._forward(inp) for inp in input.chunk(self.num_chunks)]
            x = torch.cat(out, dim=0)
        else:
            x = self._forward(input)

        if self._mean is None:
            return x
        else:
            return self.normalize_(x)

    def _forward(self,
                 input: Tensor
                 ) -> Tensor:
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](input)

    def normalize_(self,
                   input: Tensor
                   ) -> Tensor:
        # [0, 1] -> [-1, 1]
        return input.add_(- self._mean[:, None, None]).div_(self._std[:, None, None])

    def denormalize_(self,
                     input: Tensor
                     ) -> Tensor:
        # [-1, 1] -> [0, 1]
        return input.mul_(self._std[:, None, None]).add_(self._mean[:, None, None])

    @staticmethod
    def dda_operations():

        return [
            ShearX(),
            ShearY(),
            TranslateY(),
            TranslateY(),
            Rotate(),
            HorizontalFlip(),
            Invert(),
            Solarize(),
            Posterize(),
            Contrast(),
            Saturate(),
            Brightness(),
            Sharpness(),
            AutoContrast(),
            Equalize(),
        ]

    @staticmethod
    def faster_auto_augment_policy(num_sub_policies: int,
                                   temperature: float,
                                   operation_count: int,
                                   num_chunks: int,
                                   mean: Optional[torch.Tensor] = None,
                                   std: Optional[torch.Tensor] = None,
                                   ):
        if mean is None or std is None:
            mean = torch.ones(3) * 0.5
            std = torch.ones(3) * 0.5

        return Policy(nn.ModuleList(Policy.dda_operations()), num_sub_policies, temperature, operation_count,
                      num_chunks, mean=mean, std=std)
