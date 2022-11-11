"""
    Focus on a specific landmark, (not all landmarks)
"""

import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import json
import math
from tutils import tfilename
import monai.transforms as monai_transforms
from monai.transforms import SpatialPadd
import tutils.mn.data.augment.common_2d_aug as ttf
import cv2


class Cephalometric(data.Dataset):
    def __init__(self,
                 pathDataset,
                 split="Train",
                 ret_mode="all",
                 R_ratio=0.05,
                 num_landmark=19,
                 landmark_id=0,
                 size=[384, 384],
                 epoch=0,
                 do_repeat=True,
                 config=None,
                 pseudo_pth=None,
                 return_offset=True,
                 cj_brightness=0.8,
                 cj_contrast=0.6,):
        self.return_offset = return_offset
        self.num_landmark = 1
        self.Radius = int(max(size) * R_ratio)
        self.size = size
        self.epoch = epoch
        # self.pseudo = "" if pseudo is None else f"_{pseudo}"
        self.config = config
        self.pseudo_pth = pseudo_pth
        self.ret_mode = ret_mode
        self.original_size = [2400, 1935]
        self.landmark_id = landmark_id
        self.init_mask()

        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')

        self.list = list()

        if split == 'Oneshot':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 2
            end = 2
        elif split == 'Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif split == "Pseudo":
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif split == "Train-oneshot":
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 2
            end = 2
        elif split == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 170
        elif split == 'Test11':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        elif split == 'Test2':
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
        elif split == "Test1+2":
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 400
        else:
            raise ValueError

        self.transform = transforms.Compose([transforms.Resize(self.size)])

        self.aug_transform_heatmap = transforms.Compose([
            # ttf.RandomRotation(keys=['img', 'heatmap'], degrees=(-10,10)),
            # ttf.RandomHorizontalFlip(keys=['img', 'heatmap']),
            # ttf.RandomResizedCrop(keys=['img', 'heatmap'], size=self.size, scale=(0.95, 1.0)),
            # ttf.RandomAffine(keys=['img', 'heatmap'], degrees=0, translate=(0.1, 0.1)),
            ttf.ColorJitter(keys=['img',], brightness=cj_brightness, contrast=cj_contrast),
            ttf.ToTensor(keys=['img', ]),
            ttf.Normalize(keys=['img',], mean=[0], std=[1]),
        ])

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        self.split = split

        num_repeat = 9
        if split in ['Train', 'Train-oneshot'] and do_repeat:
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)
        self.do_repeat = do_repeat

        if split in ['Train-oneshot']:
            print("dataset index:", self.list)
        assert len(self) > 0, f"Dataset Path Error! Got {pathDataset}"

    def init_mask(self):
        # gen mask
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        gaussian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1
                    gaussian_mask[i][j] = math.exp(-1 * math.pow(distance, 2) / \
                                                   math.pow(self.Radius, 2))
        # gen offset
        self.mask = mask
        self.gaussian_mask = gaussian_mask

        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

    # def gen_mask(self, loc, landmark_id=0, img_shape=[384, 384]):
    #     # GT, mask, offset
    #     loc = np.array(loc)
    #     assert loc.ndim == 2
    #     num_landmark = 1
    #     y, x = self.size[0], self.size[1]
    #
    #     # heatmap[]
    #     mask = torch.zeros((num_landmark, y, x), dtype=torch.float)
    #     offset_x = torch.zeros((num_landmark, y, x), dtype=torch.float)
    #     offset_y = torch.zeros((num_landmark, y, x), dtype=torch.float)
    #     heatmap = torch.zeros((num_landmark, y, x), dtype=torch.float)
    #
    #     landmark = loc[landmark_id]
    #     margin_x_left = max(0, landmark[0] - self.Radius)
    #     margin_x_right = min(x, landmark[0] + self.Radius)
    #     margin_y_bottom = max(0, landmark[1] - self.Radius)
    #     margin_y_top = min(y, landmark[1] + self.Radius)
    #
    #     mask[landmark_id][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
    #         self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
    #     heatmap[landmark_id][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
    #         self.gaussian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
    #     offset_x[landmark_id][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
    #         self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
    #     offset_y[landmark_id][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
    #         self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
    #
    #     return mask, offset_y, offset_x

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[1-i] / self.original_size[1-i])
        return landmark

    def __getitem__(self, index):
        img, landmark_list, index = self._get_data(index)
        landmark_id = self.landmark_id
        if self.ret_mode == "all":
            return self._process_all(img, landmark_list, landmark_id=landmark_id, index=index)
        elif self.ret_mode == "heatmap_only":
            return self._process_heatmap_only(img, landmark_list, landmark_id=landmark_id, index=index)
        elif self.ret_mode == "onehot_heatmap":
            raise NotImplementedError
        #     return self._process_onehot_heatmap(img, landmark_list, index)
        else:
            raise NotImplementedError

    def _get_data(self, index):
        item = self.list[index]
        pth_img = os.path.join(self.pth_Image, item['ID']+'.bmp')
        img = self.transform(Image.open(pth_img).convert('RGB'))
        landmark_list = list()
        if self.split != 'Pseudo':
            with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
                with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                    for i in range(19):
                        landmark1 = f1.readline().split()[0].split(',')
                        landmark2 = f2.readline().split()[0].split(',')
                        landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                        landmark_list.append(self.resize_landmark(landmark))
        else:
            if self.epoch > 0:
                with open(self.config['base']['runs_dir'] + '/pseudo_labels/epoch_{0}/{1}.json'.format(self.epoch, item['ID']), 'r') as f:
                    landmark_dict = json.load(f)
            else:
                with open(tfilename(self.pseudo_pth, item['ID']+'.json'), 'r') as f:
                    landmark_dict = json.load(f)
            for key, value in landmark_dict.items():
                landmark_list.append(value)

        # print("debug: ", landmark_list)
        return img, landmark_list, index

    def _process_all(self, img, landmark_list, landmark_id=0, index=0):
        # GT, mask, offset
        y, x = img.shape[-2], img.shape[-1]
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        heatmap = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        # for i, landmark in enumerate(landmark_list):
        landmark = landmark_list[landmark_id]
        margin_x_left = max(0, landmark[0] - self.Radius)
        margin_x_right = min(x, landmark[0] + self.Radius)
        margin_y_bottom = max(0, landmark[1] - self.Radius)
        margin_y_top = min(y, landmark[1] + self.Radius)

        mask[landmark_id][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        heatmap[landmark_id][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.gaussian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        offset_x[landmark_id][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        offset_y[landmark_id][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]

        # return img, mask, offset_y, offset_x, landmark_list
        if not self.return_offset:
            return {'img': img, 'mask':mask}
        return {'img':img, 'mask':mask, 'offset_x': offset_x, 'offset_y':offset_y, 'landmark_list': landmark_list}

    def _process_heatmap_only(self, img, landmark_list, landmark_id=0, index=0):
        # GT, mask, offset
        y, x = img.size[1], img.size[0]
        heatmap = torch.zeros((self.num_landmark, y, x), dtype=torch.float32)
        landmark = landmark_list[landmark_id]
        margin_x_left = max(0, landmark[0] - self.Radius)
        margin_x_right = min(x, landmark[0] + self.Radius)
        margin_y_bottom = max(0, landmark[1] - self.Radius)
        margin_y_top = min(y, landmark[1] + self.Radius)
        heatmap[0][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.gaussian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        # print("debug ", heatmap.shape)
        data = self.aug_transform_heatmap({"img": img, "heatmap": heatmap})
        return data

    def __len__(self):
        return len(self.list)



class Test_Cephalometric(data.Dataset):
    def __init__(self,
                 pathDataset,
                 split="Test1+2",
                 size=384,
                 R_ratio=0.05,
                 landmark_id=0,
                 wo_landmarks=False,
                 ret_dict=False,
                 default_oneshot_id=114):
        self.num_landmark = 1
        self.size = size if isinstance(size, list) else [size, size]
        self.Radius = int(max(self.size) * R_ratio)
        print("The sizes are set as ", self.size)
        self.original_size = [2400, 1935]
        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')
        self.list = list()
        self.landmark_id = landmark_id

        if split == 'Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif split == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        elif split == 'Test2':
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
        elif split == 'Test1+2':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 400
        elif split == 'subtest':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 158
        elif split == 'oneshot_debug':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = default_oneshot_id + 1
            end = default_oneshot_id + 1
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0], [1]),  # 0.5
        ])

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})
        self.split = split
        self.base = 16
        self.wo_landmarks = wo_landmarks
        self.ret_dict = ret_dict

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[1-i] / self.original_size[1-i])
        return landmark

    def __getitem__(self, index, **kwargs):
        return self.return_func_0(index)

    def return_func_0(self, index):
        item = self.list[index]
        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
        if self.ret_dict:
            return {'image': item['image'], 'index':index}
        if self.wo_landmarks:
            return item['image']

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(19):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))
        return {"img":item['image'], "landmark_list": landmark_list[self.landmark_id:self.landmark_id+1], "name": item['ID'] + '.bmp'}

    def return_func_pure(self, index):
        item = self.list[index]
        pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
        im = cv2.imread(pth_img, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, self.size)

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(19):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))

        return {"img": im, "landmark_list": landmark_list, "name": item['ID'] + '.bmp'}

    def __len__(self):
        return len(self.list)