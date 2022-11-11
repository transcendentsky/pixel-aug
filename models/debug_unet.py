import torch
from .UNet3D import UNet3DEncoder2
import numpy as np


unet = UNet3DEncoder2(1,1,emb_len=64)
im = torch.rand((2,1,128,128,64))
feas = unet(im)
print(len(feas))
import ipdb; ipdb.set_trace()
