import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import json
import math
from tutils import tfilename
from .ceph_heatmap_imaug import AdapColorJitter2


class Cephalometric(data.Dataset):
    def __init__(self,
                 pathDataset,
                 mode="Train",
                 R_ratio=0.05,
                 num_landmark=19,
                 size=[384, 384],
                 epoch=0,
                 do_repeat=True,
                 config=None,
                 pseudo_pth=None,
                 return_offset=True,
                 use_gaussian_mask=False,
                 cj_brightness=0.8,
                 cj_contrast=0.6,
                 adap_mode="adap2",
                 cached_params_path=None):
        self.use_gaussian_mask = use_gaussian_mask
        self.return_offset = return_offset
        self.num_landmark = num_landmark
        self.Radius = int(max(size) * R_ratio)
        self.size = size
        self.epoch = epoch
        self.adap_mode = adap_mode
        # self.pseudo = "" if pseudo is None else f"_{pseudo}"
        self.config = config
        self.pseudo_pth = pseudo_pth

        self.original_size = [2400, 1935]
        self.init_mask()

        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')

        self.list = list()

        if mode == 'Oneshot':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 2
            end = 2
        elif mode == 'Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == "Pseudo":
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == "Train-oneshot":
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 2
            end = 2
        elif mode == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 170
        elif mode == 'Test11':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        elif mode == 'Test2':
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
        elif mode == "Test1+2":
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 400
        else:
            raise ValueError

        self.transform = transforms.Compose([transforms.Resize(self.size),
                                             transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0], [1])])
        self.resize = transforms.Resize(self.size)
        self.tensor_norm = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize([0], [1])])
        self.adapaug2 = AdapColorJitter2(cached_params_path=cached_params_path)

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        self.mode = mode

        num_repeat = 9
        if mode in ['Train', 'Train-oneshot'] and do_repeat:
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)
        self.do_repeat = do_repeat

        if mode in ['Train-oneshot']:
            print("dataset index:", self.list)
        assert len(self) > 0, f"Dataset Path Error! Got {pathDataset}"

    def init_mask(self):
        # gen mask
        standard_deviation = 10
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    if not self.use_gaussian_mask:
                        mask[i][j] = 1
                    else:
                        z = np.exp(-distance**2 / (2 * (standard_deviation ** 2)))
                        mask[i][j] = z / (np.sqrt(2 * np.pi) * standard_deviation)
        mask = mask / mask.max()

        # gen offset
        self.mask = mask

        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

    def gen_mask(self, loc, img_shape=[384, 384]):
        # GT, mask, offset
        loc = np.array(loc)
        assert loc.ndim == 2
        num_landmark = loc.shape[0]
        y, x = self.size[0], self.size[1]
        mask = torch.zeros((num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((num_landmark, y, x), dtype=torch.float)

        for i in range(num_landmark):
            landmark = loc[i]
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
        return mask, offset_y, offset_x

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[1-i] / self.original_size[1-i])
        return landmark

    def __getitem__(self, index):
        item = self.list[index]

        pth_img = os.path.join(self.pth_Image, item['ID']+'.bmp')
        img = Image.open(pth_img).convert('RGB')
        if self.adap_mode == "adap0":
            img = self.resize(img)
            img, ww, params = self.adapaug2(img)
            img = self.tensor_norm(img)
        elif self.adap_mode == "none":
            img = self.transform(img)
            ww = 0
        else:
            img = self.transform(img)
            ww = 0

        landmark_list = list()

        if self.mode != 'Pseudo':
            with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
                with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                    for i in range(self.num_landmark):
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

        # GT, mask, offset
        y, x = img.shape[-2], img.shape[-1]
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

        return {'img':img, 'mask':mask, 'offset_x': offset_x, 'offset_y':offset_y, 'landmark_list': landmark_list, "ww": ww, ""}

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    test = Cephalometric('../../dataset/Cephalometric', 'Train')
    mask, offset_y, offset_x = test.gen_mask([[10,10]] )
    import ipdb; ipdb.set_trace()