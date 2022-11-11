import torchvision.transforms as transforms
import numpy as np
import h5py, os, random, math, torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
import csv


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Abstract_basic(data.Dataset):
    def __init__(self,
                 pathDataset=None,
                 mode="Train",
                 size=[384, 384],
                 R_ratio=0.05,
                 num_landmark=37,
                 datanum=0):

        self.num_landmark = num_landmark
        self.size = size
        self.pth_Image = os.path.join(pathDataset)
        self.datanum = datanum
        self.mode = mode
        self.base = 16

        self.init_dataset()

        if self.datanum > 0:
            self.train_list = self.train_list[:datanum]
            self.landmarks_train = self.landmarks_train[:datanum]

        if mode in ["Oneshot", "Train"]:
            self.istrain = True
        elif mode in ["Test1", "Test"]:
            self.istrain = False
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.], [1.]),
        ])
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])

        self.transform_resize = transforms.Resize(self.size)
        self.transform_tensor = transforms.ToTensor()

        self.extra_aug_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.25)], p=0.5),
            transforms.ToTensor(),
            AddGaussianNoise(0., 1.),
            transforms.Normalize([0], [1]),
        ])

        # gen mask
        self.Radius = int(max(size) * R_ratio)
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1

        self.mask = mask
        self.guassian_mask = guassian_mask

        # gen offset
        self.offset_w = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_h = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_w[:, i] = self.Radius - i
            self.offset_h[i, :] = self.Radius - i
        self.offset_w = self.offset_w * self.mask / self.Radius
        self.offset_h = self.offset_h * self.mask / self.Radius

        assert len(self) > 0

    def return_img_name(self, index):
        p, name = os.path.split(self.train_list[index])
        return name

    # @abstract function
    def init_dataset(self):
        raise NotImplementedError
        # --------------------------------------------------
        self.list = [x.path for x in os.scandir(self.pth_Image) if x.name.endswith(".jpg")]
        self.list.sort()
        # print(self.list)
        self.landmarks = get_csv_content(label_path)
        self.test_list = self.list[:100]
        self.landmarks_test = self.landmarks[:100]
        self.train_list = self.list[100:]
        self.landmarks_train = self.landmarks[100:]

    # @abstract function
    def get_data(self, index):
        raise NotImplementedError
        item_path = self.train_list[index]
        img = Image.open(item_path)
        landmarks = None
        return img, landmarks

    # @abstract function
    def __len__(self):
        raise NotImplementedError
        if self.istrain:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def __getitem__(self, index):
        """
        landmark = {'index': index, 'landmark': [(x,y), (x,y), (x,y), ...]}
        """
        img, landmarks, spacing = self.get_data(index)
        return self._process(index, img, landmarks)

    def _process(self, index, img, landmarks):

        img_shape = np.array(img).shape[::-1][1:]
        print("debug: ", img_shape)
        img = self.transform(img.convert('RGB'))
        landmark_list = self.resize_landmark(landmarks,
                                             img_shape)  # [1:] for discarding the index
        # import ipdb; ipdb.set_trace()
        if not self.istrain:
            return {'img':img, 'landmark_list': landmark_list, "img_shape": img_shape, "index": index}

        y, x = img.shape[-2], img.shape[-1]
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_w = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_h = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):

            margin_x_left = max(0, landmark[0] - self.Radius)
            margin_x_right = min(x, landmark[0] + self.Radius)
            margin_y_bottom = max(0, landmark[1] - self.Radius)
            margin_y_top = min(y, landmark[1] + self.Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_w[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_w[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_h[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_h[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]

        return {'img':img, 'mask':mask, 'offset_w': offset_w, 'offset_h':offset_h,
                'landmark_list': landmark_list, "img_shape": img_shape, "index": index}

    def resize_landmark(self, landmark, img_shape):
        """
        landmark = ['index': index, 'landmark': [(h, w)), ...]]
        """
        for i in range(len(landmark)):
            # print("[Trans Debug] ", landmark[i], img_shape)
            landmark[i][0] = int(landmark[i][0] * self.size[0] / float(img_shape[0]))
            landmark[i][1] = int(landmark[i][1] * self.size[1] / float(img_shape[1]))
        return landmark



# def TestHandXray(*args, **kwargs):
#     return HandXray(mode="Test", *args, **kwargs)


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     # dataset = HandXray(pathDataset='/home1/quanquan/datasets/hand/hand/jpg/', label_path='/home1/quanquan/datasets/hand/hand/all.csv')
#     # dataset.landmarks
#     # print()
#     img = dataset.__getitem__(0)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     for data in loader:
#         import ipdb; ipdb.set_trace()

