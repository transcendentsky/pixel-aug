import torchvision.transforms as transforms
import numpy as np
import h5py, os, random, math, torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
import csv
import tutils.mn.data.augment.common_2d_aug as ttf



def get_csv_content(path):  #
    if not path.endswith('.csv'):
        raise ValueError(f"Wrong path, Got {path}")
    with open(path) as f:
        f_csv = csv.reader(f)
        res_list = []
        discard = next(f_csv)  # discard No.3142
        for i, row in enumerate(f_csv):
            res = {'index': row[0]}
            landmarks = []
            for i in range(1, len(row), 2):
                # print(f"{i}")
                # landmarks += [(row[i], row[i+1])] #
                landmarks += [[int(row[i]), int(row[i + 1])]]  # change _tuple to _list
            res['landmark'] = landmarks
            res_list += [res]
        return res_list


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class HandXray(data.Dataset):
    def __init__(self,
                 pathDataset='/home/quanquan/hand/hand/',
                 split="Train",
                 size=[384, 384],
                 R_ratio=0.05,
                 num_landmark=37,
                 datanum=0,
                 num_repeat=1,
                 ret_mode="all",
                 cj_brightness=0.15,
                 cj_contrast=0.25,):

        if split in ["Oneshot", "Train", "pseudo"]:
            self.istrain = True
        elif split in ["Test1", "Test"]:
            self.istrain = False
        else:
            raise NotImplementedError
            self.istrain = True
            print(f"Using split = {split}")

        self.num_landmark = num_landmark
        self.size = size
        self.pth_Image = os.path.join(pathDataset, "jpg")
        self.datanum = datanum
        self.ret_mode = ret_mode
        self.num_repeat = num_repeat

        self.list = [x.path for x in os.scandir(self.pth_Image) if x.name.endswith(".jpg")]
        self.list.sort()
        # print(self.list)
        label_path = os.path.join(pathDataset, "all.csv")
        self.landmarks = get_csv_content(label_path)
        self.test_list = self.list[:300]
        self.landmarks_test = self.landmarks[:300]
        self.train_list = self.list[300:]
        self.landmarks_train = self.landmarks[300:]

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.], [1.]),
        ])

        self.transform_resize = transforms.Resize(self.size)
        self.transform_tensor = transforms.ToTensor()

        # transforms.RandomChoice(transforms)
        # transforms.RandomApply(transforms, p=0.5)
        # transforms.RandomOrder()
        self.extra_aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
                transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast)], p=0.5),
            transforms.ToTensor(),
            # AddGaussianNoise(0., 1.),
            transforms.Normalize([0], [1]),
        ])
        self.aug_transform_heatmap = transforms.Compose([
            # ttf.RandomRotation(keys=['img', 'heatmap'], degrees=(-10,10)),
            # ttf.RandomHorizontalFlip(keys=['img', 'heatmap']),
            # ttf.RandomResizedCrop(keys=['img', 'heatmap'], size=self.size, scale=(0.95, 1.0)),
            # ttf.RandomAffine(keys=['img', 'heatmap'], degrees=0, translate=(0.1, 0.1)),
            ttf.ColorJitter(keys=['img', ], brightness=0.15, contrast=0.25),
            ttf.ToTensor(keys=['img', ]),
            ttf.Normalize(keys=['img', ], mean=[0], std=[1]),
        ])
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])
        self.split = split
        self.base = 16

        # gen mask
        self.Radius = int(max(size) * R_ratio)
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        gaussian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1
                    gaussian_mask[i][j] = math.exp(-1 * math.pow(distance, 2) / \
                                                   math.pow(self.Radius, 2))

        self.mask = mask
        self.gaussian_mask = gaussian_mask

        # gen offset
        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

        if self.istrain:
            # self.loading_list = np.arange(0, len(self.train_list)*num_repeat)
            self.loading_list = self.train_list
            self.loading_landmark_list = self.landmarks_train
            if self.datanum > 0:
                self.loading_list = self.loading_list[:datanum]
                self.loading_landmark_list = self.loading_landmark_list[:datanum]
            self.real_len = len(self.loading_list)
        else:
            self.loading_list = self.test_list
            self.loading_landmark_list = self.landmarks_test
            if self.datanum > 0:
                self.loading_list = self.loading_list[:datanum]
                self.loading_landmark_list = self.loading_landmark_list[:datanum]
            self.real_len = len(self.loading_list)

        assert len(self) > 0

    def return_img_name(self, index):
        p, name = os.path.split(self.train_list[index])
        return name

    def return_name(self, index):
        return self.return_img_name(index)

    def __getitem__(self, index):
        index = index % self.real_len
        img, landmark_list, index, img_shape = self._get_data(index)
        if self.ret_mode == "all":
            return self._process_all(img, landmark_list, index, img_shape)
        elif self.ret_mode == "heatmap_only":
            return self._process_heatmap_only(img, landmark_list, index, img_shape)
        # elif self.ret_mode == "onehot_heatmap":
        #     return self._process_onehot_heatmap(img, landmark_list, index, img_shape)
        elif self.ret_mode == "no_process":
            return {'img':img, 'landmark_list': landmark_list, "index": index, "img_shape": img_shape}
        else:
            raise NotImplementedError

    def _get_data(self, index, return_ori=False):
        item_path = self.loading_list[index]
        ori_img = Image.open(item_path)
        img_shape = np.array(ori_img).shape[::-1]  # shape: (w, h)
        # if self.transform is not None:
        item = self.transform(ori_img.convert('RGB'))
        # import ipdb; ipdb.set_trace()
        landmark_list = self.resize_landmark(self.loading_landmark_list[index]['landmark'],
                                        img_shape)  # [1:] for discarding the index
        if return_ori:
            return item, landmark_list, index, img_shape, ori_img
        return item, landmark_list, index, img_shape

    def _process_all(self, item, landmark_list, index, img_shape):
        """
        landmark = {'index': index, 'landmark': [(x,y), (x,y), (x,y), ...]}
        """
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

        return {'img':item, 'mask':mask, 'offset_x': offset_x, 'offset_y':offset_y,
                'landmark_list': landmark_list, "img_shape": img_shape, "index": index}

    def _process_heatmap_only(self, img, landmark_list, index, img_shape):
        # GT, mask, offset
        # print(img.shape, img.size)
        y, x = img.shape[-2], img.shape[-1]
        heatmap = torch.zeros((self.num_landmark, y, x), dtype=torch.float32)

        for i, landmark in enumerate(landmark_list):
            margin_x_left = max(0, landmark[0] - self.Radius)
            margin_x_right = min(x, landmark[0] + self.Radius)
            margin_y_bottom = max(0, landmark[1] - self.Radius)
            margin_y_top = min(y, landmark[1] + self.Radius)
            heatmap[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.gaussian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        # print("debug ", heatmap.shape, img.shape)
        # data = self.aug_transform_heatmap({"img": img, "heatmap": heatmap})
        img = self.extra_aug_transform(img)
        data = {"img": img, "heatmap": heatmap}
        return data

    def resize_landmark(self, landmark, img_shape):
        """
        landmark = ['index': index, 'landmark': [(x,y), (x,y), (x,y), ...]]
        """
        for i in range(len(landmark)):
            # print("[Trans Debug] ", landmark[i], img_shape)
            landmark[i][0] = int(landmark[i][0] * self.size[1] / float(img_shape[0]))
            landmark[i][1] = int(landmark[i][1] * self.size[0] / float(img_shape[1]))
        return landmark

    def __len__(self):
        return len(self.loading_list) * self.num_repeat


def TestHandXray(*args, **kwargs):
    return HandXray(split="Test", *args, **kwargs)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tutils import torchvision_save
    dataset = HandXray(pathDataset='/home1/quanquan/datasets/hand/hand/', split="Train", ret_mode="heatmap_only")
    # dataset.landmarks
    # print()
    # img = dataset.__getitem__(0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in loader:
        img = data['img']
        heatmap = data['heatmap']
        torchvision_save(heatmap[:, :1, :, :], "./.tmp/heatmap_hand.png")
        torchvision_save(img, "./.tmp/img_hand.png")
        import ipdb; ipdb.set_trace()

