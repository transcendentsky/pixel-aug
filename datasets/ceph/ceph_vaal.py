from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy


def ceph_transformer():
    return torchvision.transforms.Compose([
        transforms.RandomResizedCrop(192),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                             std=[0.5, 0.5, 0.5]),
    ])



class Ceph(Dataset):
    def __init__(self, path, test=False):
        self.imagenet = datasets.ImageFolder(root=path, transform=ceph_transformer())
        assert len(self) >= 150, f"datalen Got {len(self)}"
        self.test = test

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        # print("debug: ", index)
        data, target = self.imagenet[index]

        if self.test:
            return data, index
        # data (img), index (input index), target (No. of subfolder)
        # return data, index, target
        return data, target, index

    def __len__(self):
        return len(self.imagenet)
