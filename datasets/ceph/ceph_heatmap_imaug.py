"""
    imbalanced aug

    Mutual info:
    0.8334361740522674 0.236187758371206
[0.58855864 0.9689842  0.88423108 0.41595766 0.93467499 0.87520715
 0.75355031 0.69097756 0.70706193 0.9617875  0.95781572 0.96094506
 0.98639255 0.89073955 0.96499311 0.71407015 0.93329412 0.91891464
 0.7271314 ]
[0.26388173 0.12006105 0.12696776 0.31245908 0.10358765 0.20422783
 0.21599622 0.18820669 0.19157507 0.13725611 0.11465392 0.12241823
 0.11602296 0.19213057 0.09218982 0.18752875 0.1210562  0.1017045
 0.28287337]

    Mutual info (self):
    2.781115538153201 0.6866188393380371
[2.51138927 3.18064627 3.1936682  2.11334156 3.26182041 2.72902131
 2.16312098 1.94328915 2.01532239 3.26112403 3.20813622 3.16552408
 3.12697808 2.64613941 3.25124197 1.97482131 3.206975   3.22484813
 2.66378744]
[0.46076691 0.30574453 0.2845213  0.48100281 0.24713011 0.75816679
 0.75060892 0.65067037 0.6793318  0.23704422 0.36165119 0.38236706
 0.30267382 0.60297583 0.24602729 0.60459519 0.21163927 0.24409171
 0.47145102]

    Ratio: MI / self-MI
    0.30255598616518015 0.07071514569084383
[0.22228368 0.3052235  0.27727049 0.17564076 0.28687285 0.33020751
 0.35978205 0.36509693 0.36119425 0.29635247 0.30038729 0.30547126
 0.31645546 0.3409684  0.29784311 0.36810754 0.29157507 0.28518113
 0.26264999]
[0.08518929 0.03811567 0.03536874 0.11798476 0.0276701  0.04630233
 0.04844656 0.04309345 0.04499098 0.04691302 0.03214592 0.03304589
 0.03321996 0.03880205 0.03040845 0.03984614 0.03734166 0.026325
 0.08700717]

    Entropy:
    4.592848378070493 0.5059876207290258
[4.32922599 4.94955578 4.9402223  4.13369406 5.02937445 4.46131056
 4.12397278 4.07084514 4.08404715 4.8864486  4.87044201 4.79871701
 4.7067016  4.29701387 4.94452249 4.05091702 5.03718015 5.01795204
 4.53197619]
[0.32104817 0.36015503 0.25001072 0.34826474 0.22153489 0.53958994
 0.44612867 0.39377246 0.4095514  0.27827981 0.33560316 0.348146
 0.33046936 0.41912717 0.26862784 0.38550231 0.17205685 0.20877866
 0.32103766]

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
# import monai.transforms as monai_transforms
# from monai.transforms import SpatialPadd
import tutils.mn.data.augment.common_2d_aug as ttf
from datasets.ceph.ceph_ssl import Test_Cephalometric
from torchvision.transforms import functional as F
import copy
from utils.entropy_loss import get_guassian_heatmaps_from_ref
import pandas as pd


mi_mean = [0.58855864,0.9689842,0.88423108,0.41595766,0.93467499,0.87520715
            ,0.75355031,0.69097756,0.70706193,0.9617875,0.95781572,0.96094506
            ,0.98639255,0.89073955,0.96499311,0.71407015,0.93329412,0.91891464
            ,0.7271314]
mi_std = [0.26388173,0.12006105,0.12696776,0.31245908,0.10358765,0.20422783
            ,0.21599622,0.18820669,0.19157507,0.13725611,0.11465392,0.12241823
            ,0.11602296,0.19213057,0.09218982,0.18752875,0.1210562, 0.1017045
            ,0.28287337]

miself_mean = [2.51138927,3.18064627,3.1936682,2.11334156,3.26182041,2.72902131
            ,2.16312098,1.94328915,2.01532239,3.26112403,3.20813622,3.16552408
            ,3.12697808,2.64613941,3.25124197,1.97482131,3.206975,3.22484813
            ,2.66378744]
miself_std = [0.46076691,0.30574453,0.2845213,0.48100281,0.24713011,0.75816679
,0.75060892,0.65067037,0.6793318, 0.23704422,0.36165119,0.38236706
,0.30267382,0.60297583,0.24602729,0.60459519,0.21163927,0.24409171
,0.47145102]

entr_mean = [4.32922599,4.94955578,4.9402223,4.13369406,5.02937445,4.46131056
            ,4.12397278,4.07084514,4.08404715,4.8864486, 4.87044201,4.79871701
            ,4.7067016, 4.29701387,4.94452249,4.05091702,5.03718015,5.01795204
            ,4.53197619]
entr_std = [0.32104817,0.36015503,0.25001072,0.34826474,0.22153489,0.53958994
,0.44612867,0.39377246,0.4095514, 0.27827981,0.33560316,0.348146
,0.33046936,0.41912717,0.26862784,0.38550231,0.17205685,0.20877866
,0.32103766]

mre = [1.399, 1.5526,1.7562,2.9915 , 1.7145, 1.9027,
       1.8260, 1.3118, 1.1585, 4.4261, 2.2153, 2.8174,
       1.5348, 1.9849, 1.3075, 1.8713,1.5415, 3.7358,
       2.3073]
mre2 =[1.6767,1.7246,2.2157,2.5067,1.9819,2.2091,
       1.7173,1.3327,1.6392,2.2740,1.6451,1.7791,
       1.5442,1.7209,1.8208,2.9706,1.8948,1.9768,
       2.1830]

class Cephalometric(data.Dataset):
    def __init__(self,
                 pathDataset,
                 mode="Train",
                 ret_mode="all",
                 R_ratio=0.05,
                 num_landmark=19,
                 size=[384, 384],
                 epoch=0,
                 do_repeat=True,
                 config=None,
                 pseudo_pth=None,
                 adap_mode=2,
                 return_offset=False):
        self.return_offset = return_offset
        self.num_landmark = num_landmark
        self.Radius = int(max(size) * R_ratio)
        self.size = size
        self.epoch = epoch
        # self.pseudo = "" if pseudo is None else f"_{pseudo}"
        self.config = config
        self.pseudo_pth = pseudo_pth
        self.ret_mode = ret_mode
        self.original_size = [2400, 1935]
        self.adap_mode = adap_mode
        self.init_mask()

        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')
        self.list = list()
        self.real_len = 150

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

        self.transform = transforms.Compose([transforms.Resize(self.size)])
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0], [1])
        self.adapaug2 = AdapColorJitter2()
        # 0.5941818181818173 1.5424181818181808 0.4595151515151508 1.7238303030303037
        self.adapaug_transform = AdapColorJitter(brightness=[0.5941818181818173, 1.5424181818181808], contrast=[0.4595151515151508, 1.7238303030303037])
        self.aug_transform_heatmap = transforms.Compose([
            # ttf.ToPIL(keys=['img',]),
            ttf.RandomAffine(keys=['img', 'heatmap'], degrees=0, translate=(0.1, 0.1)),
            ttf.ToTensor(keys=['img', ]),
            ttf.Normalize(keys=['img',], mean=[0], std=[1]),
        ])

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

    def gen_mask(self, loc, img_shape=[384, 384]):
        # GT, mask, offset
        loc = np.array(loc)
        assert loc.ndim == 2
        num_landmark = loc.shape[0]
        y, x = self.size[0], self.size[1]

        # heatmap[]
        mask = torch.zeros((num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((num_landmark, y, x), dtype=torch.float)
        heatmap = torch.zeros((num_landmark, y, x), dtype=torch.float)

        for i in range(num_landmark):
            landmark = loc[i]
            margin_x_left = max(0, landmark[0] - self.Radius)
            margin_x_right = min(x, landmark[0] + self.Radius)
            margin_y_bottom = max(0, landmark[1] - self.Radius)
            margin_y_top = min(y, landmark[1] + self.Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            heatmap[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.gaussian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
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
        img, landmark_list, index = self._get_data(index)
        if self.ret_mode == "all":
            return self._process_all(img, landmark_list, index)
        elif self.ret_mode == "heatmap_only":
            return self._process_heatmap_only(img, landmark_list, index)
        elif self.ret_mode == "onehot_heatmap":
            return self._process_onehot_heatmap(img, landmark_list, index)
        else:
            raise NotImplementedError

    def _get_data(self, index):
        item = self.list[index]
        pth_img = os.path.join(self.pth_Image, item['ID']+'.bmp')
        img = self.transform(Image.open(pth_img).convert('RGB'))

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

        return img, landmark_list, index

    def _process_all(self, img, landmark_list, index):
        # GT, mask, offset
        y, x = img.shape[-2], img.shape[-1]
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        heatmap = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):
            margin_x_left = max(0, landmark[0] - self.Radius)
            margin_x_right = min(x, landmark[0] + self.Radius)
            margin_y_bottom = max(0, landmark[1] - self.Radius)
            margin_y_top = min(y, landmark[1] + self.Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            heatmap[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.gaussian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]

        # return img, mask, offset_y, offset_x, landmark_list
        if not self.return_offset:
            return {'img': img, 'mask':mask}
        return {'img':img, 'mask':mask, 'offset_x': offset_x, 'offset_y':offset_y, 'landmark_list': landmark_list}

    def _process_heatmap_only(self, img, landmark_list, index):
        # GT, mask, offset
        y, x = img.size[1], img.size[0]
        heatmap = torch.zeros((self.num_landmark, y, x), dtype=torch.float32)

        for i, landmark in enumerate(landmark_list):
            margin_x_left = max(0, landmark[0] - self.Radius)
            margin_x_right = min(x, landmark[0] + self.Radius)
            margin_y_bottom = max(0, landmark[1] - self.Radius)
            margin_y_top = min(y, landmark[1] + self.Radius)
            heatmap[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.gaussian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        # print("debug ", heatmap.shape)

        if self.adap_mode == "1":
            img = self.normalize(self.totensor(img))
            img = self.adapaug_transform(img, index=index % self.real_len)
        else:
            img, ww, params = self.adapaug2(img)
            img = self.totensor(img)
            img = self.normalize(img)
            return {"img": img.float(), "heatmap": heatmap.float(), "ww": ww}
        # data = self.aug_transform_heatmap({"img": img, "heatmap": heatmap})
        data = {"img": img.float(), "heatmap": heatmap.float()}
        return data

    @staticmethod
    def im_aug(im):
        pass

    def _process_onehot_heatmap(self, img, landmark_list, index):
        y, x = img.size[1], img.size[0]
        heatmap = torch.zeros((self.num_landmark, y, x), dtype=torch.float32)
        for i, landmark in enumerate(landmark_list):
            heatmap[i][landmark[1], landmark[0]] = 1
        data = self.aug_transform_heatmap({"img": img, "heatmap": heatmap})
        return data

    def __len__(self):
        return len(self.list)


class AdapColorJitter(transforms.ColorJitter):
    def __init__(self, split='Train', *args, **kwargs):
        # if not hasattr(self, 'testset'):
        self.testset = Test_Cephalometric(mode=split)
        self.landmarks = [self.testset.ref_landmarks(i) for i in range(len(self.testset))]
        self.probmaps = [self.get_probs_from_lm(lm) for lm in self.landmarks]
        self.weights = self.get_weights()
        super().__init__(*args, **kwargs)

    # def forward(self, ori_img, weights=[], probmaps=None):
    def forward(self, ori_img, index):
        weights = self.weights
        probmaps = self.probmaps[index]

        assert isinstance(weights, (np.ndarray, list, type(None))), f"Got weights: {type(weights)}"
        # assert len(weights) > 0, f"Got "

        if weights is None or len(weights) == 0:
            raise NotImplementedError
        else:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            img_list = []
            weights = weights.tolist()
            weights.append(1.0)
            probmaps.append(np.ones_like(probmaps[0])*0.1)
            # print("AdapCJ debug: ", weights, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)

            for wi in weights:
                img = copy.copy(ori_img)
                # print("debug copy", wi, ori_img.sum())
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = F.adjust_brightness(img, brightness_factor * wi)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = F.adjust_contrast(img, contrast_factor * wi)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = F.adjust_saturation(img, saturation_factor * wi)
                    elif fn_id == 3 and hue_factor is not None:
                        img = F.adjust_hue(img, hue_factor * wi)
                img_list.append(img)
            # return img_list
            if probmaps is not None:
                return self.im_fusion(img_list, probmaps)
            else:
                raise NotImplementedError

    def get_weights(self, cached_params_path='cache/optimal_brct_im.npy'):
        self.cached_params_path = cached_params_path
        self.cached_params = np.load(cached_params_path, allow_pickle=True).tolist()
        self.pd_params = pd.DataFrame.from_dict(self.cached_params)
        aug_list = []
        for index in range(19):
            d = self.cached_params[index]
            cj_br, cj_ct = [d['cj_br_floor'], d['cj_br_ceil']], [d['cj_ct_floor'], d['cj_ct_ceil']]
            aug_list.append([cj_br, cj_ct])

        # --- Adjust Error
        aug_list = (np.array(aug_list) - 1) * 0.8 + 1
        # mean_value = aug_list.mean()
        mean_value = aug_list / aug_list.mean(axis=0)
        w = mean_value.mean(axis=-1).mean(axis=-1)
        return w

    def get_landmarks(self, index, split="Train"):
        return self.landmarks[index]

    @staticmethod
    def im_fusion(img_list, probmaps):
        prob_sum = np.array(probmaps).sum(axis=0)
        img2_list = [img * prob / prob_sum for img, prob in zip(img_list, probmaps)]
        img = np.array(img2_list, dtype=object).sum()
        return img

    @staticmethod
    def get_probs_from_lm(lms, size=(384,384), kernel_size=192, sharpness=0.2):
        prob_list = []
        for lm in lms:
            prob_map = get_guassian_heatmaps_from_ref(landmarks=[lm], num_classes=1, \
                                                   image_shape=size, kernel_size=kernel_size,
                                                   sharpness=sharpness)  # shape: (19, 800, 640)
            prob_list.append(prob_map)
        return prob_list


class AdapColorJitter2(transforms.ColorJitter):
    def __init__(self, cached_params_path='cache/optimal_brct_im.npy'):
        self.params_thresholds = self.get_thresholds(cached_params_path=cached_params_path)
        self.aug_percent = self.get_aug_percent(self.params_thresholds)
        p1 = self.params_thresholds.min(axis=0)
        p2 = self.params_thresholds.max(axis=0)
        br = [p1[0][0], p2[0][1]]
        ct = [p1[1][0], p2[1][1]]
        super().__init__(brightness=br, contrast=ct)

    def forward(self, img):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        cc = self._check_thresholds(self.params_thresholds, (brightness_factor, contrast_factor))
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img, cc, (brightness_factor, contrast_factor)

    def get_thresholds(self, cached_params_path='cache/optimal_brct_im.npy'):
        self.cached_params_path = cached_params_path
        self.cached_params = np.load(cached_params_path, allow_pickle=True).tolist()
        self.pd_params = pd.DataFrame.from_dict(self.cached_params)
        aug_list = []
        for index in range(19):
            d = self.cached_params[index]
            cj_br, cj_ct = [d['cj_br_floor'], d['cj_br_ceil']], [d['cj_ct_floor'], d['cj_ct_ceil']]
            aug_list.append([cj_br, cj_ct])
        # Adjust Errors
        # aug_list = (np.array(aug_list) - 1) * 0.8 + 1
        aug_list = np.array(aug_list)
        return aug_list

    def get_aug_percent(self, params):
        p1 = params.min(axis=0)
        p2 = params.max(axis=0)
        br = [p1[0][0], p2[0][1]]
        ct = [p1[1][0], p2[1][1]]
        br_p = np.array(params[:, 0, 1] - params[:, 0, 0]) / (br[1] - br[0])
        ct_p = np.array(params[:, 1, 1] - params[:, 1, 0]) / (ct[1] - ct[0])
        tt_p = br_p * ct_p
        return tt_p

    def _check_thresholds(self, thresholds, params):
        brightness_factor, contrast_factor = params
        d = []
        for thres in thresholds:
            br_thres, ct_thres = thres
            d.append(_check_in(brightness_factor, br_thres) and _check_in(contrast_factor, ct_thres))
        return np.array(d)


def _check_in(factor, thres):
    return (factor >= thres[0] and factor <= thres[1])


if __name__ == '__main__':
    # aug_fn = AdapColorJitter(brightness=[0.6, 1.4])
    # img = torch.ones((384,384))
    # probmaps_lm = (torch.rand((19,2)) * 384).int()
    # probmaps = aug_fn.get_probs_from_lm(probmaps_lm)
    # weights = np.arange(0,1.9, 0.1)
    # print(weights)

    # aug_img = aug_fn(img, weights, probmaps)

    aug_fn = AdapColorJitter2()
    print
    # data = torch.ones((3,384,384))
    # cc_list = []
    # for i in range(10000):
    #     img, cc, params = aug_fn(data)
    #     cc_list.append(cc)
    # cc_list = np.array(cc_list)
    # print(cc_list.mean(axis=0))
    # aug_list = aug_fn.get_thresholds()
    import ipdb; ipdb.set_trace()





