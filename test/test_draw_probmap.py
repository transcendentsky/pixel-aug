import numpy as np
from tutils import torchvision_save, tfilename
import torch


print("[*] Accessing prob maps!")

path_dir = '/home1/quanquan/datasets/Cephalometric/prob2/train/'

for i in [1, 2, 10]:
    path = path_dir + f'{i}.npy'
    prob_map = np.load(path)
    torchvision_save(torch.Tensor(prob_map.copy() / prob_map.max()),
                 tfilename(path_dir, f"{i}.png"))
