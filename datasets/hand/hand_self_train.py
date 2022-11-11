import torchvision.transforms as transforms
import numpy as np
import h5py, os, random, math, torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
import csv
import json
from tutils import tfilename, tdir
from .hand_basic import HandXray as BasicHandXray


class HandXray(BasicHandXray):
    def __init__(self, pseudo_path=None, *args, **kwargs):
        self.pseudo_path = pseudo_path
        super(HandXray, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        landmark = {'index': index, 'landmark': [(x,y), (x,y), (x,y), ...]}
        """
        index = index % self.real_len
        item_path = self.train_list[index]
        ori_img = Image.open(item_path)
        img_shape = np.array(ori_img).shape[::-1]  # shape: (y, x) or (long, width)
        item = self.transform(ori_img.convert('RGB'))

        if self.split != 'pseudo':
            landmark_list = self.resize_landmark(self.landmarks_train[index]['landmark'],
                                                 img_shape)  # [1:] for discarding the index
        else:
            p, name = os.path.split(item_path)
            landmark_list = []
            with open(tfilename(self.pseudo_path, f"{name[:-4]}.json"), 'r') as f:
                landmark_dict = json.load(f)
            for key, value in landmark_dict.items():
                landmark_list.append(value)

        y, x = item.shape[-2], item.shape[-1]
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):

            margin_x_left = max(0, landmark[0] - self.Radius)
            margin_x_right = min(x, landmark[0] + self.Radius)
            margin_y_bottom = max(0, landmark[1] - self.Radius)
            margin_y_top = min(y, landmark[1] + self.Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]

        return {'img': item, 'mask': mask, 'offset_x': offset_x, 'offset_y': offset_y,
                'landmark_list': landmark_list, "img_shape": img_shape, "index": index}



def TestHandXray(*args, **kwargs):
    return HandXray(mode="Test", *args, **kwargs)


if __name__ == '__main__':
    dataset = HandXray(pathDataset='/home1/quanquan/datasets/hand/hand/jpg/', label_path='/home1/quanquan/datasets/hand/hand/all.csv')
    img = dataset.__getitem__(0)
    import ipdb; ipdb.set_trace()