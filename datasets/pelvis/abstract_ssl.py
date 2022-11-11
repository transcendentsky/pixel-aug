"""
    from cas-qs/datasets/data_loader_gp.py
"""
import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
from datasets.augment import cc_augment
from utils.entropy_loss import get_guassian_heatmaps_from_ref


def to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8).transpose(1, 2, 0))
    return images


def augment_patch(tensor, aug_transform):
    image = to_PIL(tensor)
    aug_image = aug_transform(image)
    return aug_image


class DatasetTrain(data.Dataset):
    def __init__(self,
                 pathDataset=None,
                 mode="Train",
                 size=384,
                 patch_size=192,
                 num_repeat=10):

        self.size = size if isinstance(size, list) else [size, size]
        # self.original_size = [2400, 1935]
        self.mode = mode
        self.base = 16
        self.pth_Image = None # os.path.join(pathDataset, 'pngs')
        self.pth_label = None # os.path.join(pathDataset, 'labels')
        self.patch_size = patch_size
        self.retfunc = 1

        transform_list = [
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0], [1]),
        ]
        self.transform = transforms.Compose(transform_list)

        transform_list2 = [
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.aug_transform = transforms.Compose(transform_list2)

    def resize_landmark(self, landmark, img_shape):
        """
        landmark = ['index': index, 'landmark': [(x,y), (x,y), (x,y), ...]]
        """
        for i in range(len(landmark)):
            # print("[Trans Debug] ", landmark[i], img_shape)
            landmark[i][0] = int(landmark[i][0] * self.size[0] / float(img_shape[0]))
            landmark[i][1] = int(landmark[i][1] * self.size[1] / float(img_shape[1]))
        return landmark

    def get_img_from_index(self, index):
        item = self.list[index]
        Image.open(pth_img).convert('RGB')

        return img

    def retfunc1(self, index):
        """
        New Point Choosing Function without prob map
        """
        np.random.seed()
        img = self.get_img_from_index(index)
        img_shape = np.array(img).shape[::-1]
        if self.transform != None:
            img = self.transform(img)
        pad_scale = 0.05
        padding = int(pad_scale * self.size[0])
        patch_size = self.patch_size
        raw_w = np.random.randint(int(pad_scale * self.size[0]), int((1 - pad_scale) * self.size[0]))
        raw_h = np.random.randint(int(pad_scale * self.size[1]), int((1 - pad_scale) * self.size[1]))

        b1_left = 0
        b1_top = 0
        b1_right = self.size[0] - patch_size
        b1_bot = self.size[1] - patch_size
        b2_left = raw_w - patch_size + 1
        b2_top = raw_h - patch_size + 1
        b2_right = raw_w
        b2_bot = raw_h
        b_left = max(b1_left, b2_left)
        b_top = max(b1_top, b2_top)
        b_right = min(b1_right, b2_right)
        b_bot = min(b1_bot, b2_bot)
        left = np.random.randint(b_left, b_right)
        top = np.random.randint(b_top, b_bot)

        margin_w = left
        margin_h = top
        cimg = img[:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size]
        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_w, chosen_h = raw_w - margin_w, raw_h - margin_h

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_h, chosen_w] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size

        return {'raw_imgs': img, 'crop_imgs': crop_imgs,
                'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'chosen_loc': torch.LongTensor([chosen_h, chosen_w]), 'index': index}

    def __getitem__(self, index):
        if self.retfunc == 1:
            return self.retfunc1(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.list)


class DatasetTest(data.Dataset):
    def __init__(self,
                 pathDataset,
                 mode="Test",
                 size=[384, 384]
                 ):

        self.num_landmark = 19
        self.size = size
        self.mode = mode
        self.base = 16
        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')

        self.list = list()

        self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')

        transform_list = [
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0], [1]),
        ]
        self.transform = transforms.Compose(transform_list)

    def resize_landmark(self, landmark, img_shape):
        """
        landmark = ['index': index, 'landmark': [(x,y), (x,y), (x,y), ...]]
        """
        for i in range(len(landmark)):
            # print("[Trans Debug] ", landmark[i], img_shape)
            landmark[i][0] = int(landmark[i][0] * self.size[0] / float(img_shape[0]))
            landmark[i][1] = int(landmark[i][1] * self.size[1] / float(img_shape[1]))
        return landmark

    def __getitem__(self, index):
        return self.retfunc_old(index)

    def retfunc_old(self, index):
        np.random.seed()
        item = self.list[index]

        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            img = self.transform(Image.open(pth_img).convert('RGB'))

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID'] + '.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID'] + '.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5 * (int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))

        if self.mode not in ['Oneshot', 'Fewshots']:
            # print("??, ", img.shape)
            return img, landmark_list

        template_patches = torch.zeros([self.num_landmark, 3, 192, 192])
        for id, landmark in enumerate(landmark_list):
            left = min(max(landmark[0] - 96, 0), self.size[0] - 192)
            bottom = min(max(landmark[1] - 96, 0), self.size[0] - 192)
            template_patches[id] = img[:, bottom:bottom + 192, left:left + 192]
            landmark_list[id] = [landmark[0] - left, landmark[1] - bottom]
        return img, landmark_list, template_patches

    def __len__(self):
        return len(self.list)

