import torchvision.transforms as transforms
import numpy as np
import h5py, os, random, math, torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
import csv
from tutils import tfilename

from datasets.abstract.abstract_basic import Abstract_basic


class Pelvis(Abstract_basic):
    def __init__(self, pathDataset=None,
                 mode="Train",
                 size=[384, 384],
                 R_ratio=0.05,
                 num_landmark=10,
                 datanum=0):
        self.path = pathDataset
        self.original_size = [2544, 3056]
        self.size = size
        super(Pelvis, self).__init__(pathDataset=pathDataset,
                                     mode=mode,
                                     size=size,
                                     R_ratio=R_ratio,
                                     num_landmark=num_landmark,
                                     datanum=datanum)


    # @abstract function
    def init_dataset(self):
        print("Initilize Dataset")
        self.path_landmarks = [x.name for x in os.scandir(tfilename(self.path, "labels")) if x.name.endswith(".txt")]
        self.path_landmarks.sort()

        self.path_landmarks_train = self.path_landmarks[:100]
        self.path_landmarks_test  = self.path_landmarks[100:]

    # @abstract function
    def get_data(self, index):
        # return landmarks:(w, h) or (x, y)
        spacing = 1.0
        if self.mode in ['Train']:
            label_path = self.path_landmarks_train[index]
        elif self.mode in ['Test']:
            label_path = self.path_landmarks_test[index]

        item_path = tfilename(self.path, "pngs", label_path[:-4] + ".png")
        img = Image.open(item_path).convert("RGB")
        landmarks = self.get_one_txt_content(label_path)
        # print("label_path: ", label_path)
        # print("img___path: ", item_path)

        # return landmarks:(w, h) or (x, y)
        return img, landmarks, spacing

    def __len__(self):
        if self.mode in ['Train']:
            return len(self.path_landmarks_train)
        elif self.mode in ['Test']:
            return len(self.path_landmarks_test)
        else:
            raise NotImplementedError

    def get_one_txt_content(self, path):
        # return (h,w)
        path = tfilename(self.path, "labels", path)
        lms = []
        with open(path, "r") as f:
            num_lm = int(f.readline())
            assert num_lm == 10
            for i in range(num_lm):
                s = f.readline()
                s = s.split(" ")
                lm = [float(s[1]), float(s[0])]
                lms.append(lm)
        return lms

    def get_txt_content(self, paths):
        # return [(h, w)]
        all_lms = []
        for path in paths:
            lms = self.get_one_txt_content(path)
            all_lms.append(lms)
        return all_lms

    def __getitem__(self, index):
        """
        landmark = {'index': index, 'landmark': [(x,y), (x,y), (x,y), ...]}
        """
        img, landmarks, spacing = self.get_data(index)
        return self._process(index, img, landmarks)

    def _process(self, index, img, landmarks):

        img_shape = np.array(img).shape[::-1][1:]
        img = self.transform(img.convert('RGB'))
        landmark_list = [[lm[1]*self.size[1], lm[0]*self.size[0]] for lm in landmarks]
        landmark_list = np.array(landmark_list).round().astype(int)

        if not self.istrain:
            return {'img':img, 'landmark_list': landmark_list, "img_shape": img_shape, "index": index}

        y, x = img.shape[-2], img.shape[-1]
        assert y == 384, f"Got not 384? Got y={y}"
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


if __name__ == '__main__':
    from utils.draw_landmarks import visualize
    from torch.utils.data import DataLoader
    dataset = Pelvis("/home1/quanquan/datasets/pelvis/", mode="Train")
    print("dataset len: ", len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in loader:
        # data = dataset.__getitem__(0)
        print(data['landmark_list'])
        # import ipdb; ipdb.set_trace()
        print("index dataloader: ", data['index'])
        index = data['index']
        image = visualize(data['img'], data['landmark_list'][0], data['landmark_list'][0])
        image.save(tfilename(f"tmp/pelvis/ttttmp_id_{index}.png"))
        # test_landmarks = [[50,200]]
        # image_test = visualize(data['img'], test_landmarks, test_landmarks)
        # image_test.save("tmp/testtmp.png")
        # lm = data['landmark_list']
        # print(data['landmark_list'])
        # import ipdb; ipdb.set_trace()