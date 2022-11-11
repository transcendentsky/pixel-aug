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
from .augment import cc_augment
from utils.entropy_loss import get_guassian_heatmaps_from_ref


def to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8).transpose(1, 2, 0))
    return images


def augment_patch(tensor, aug_transform):
    image = to_PIL(tensor)
    # image.save('raw.jpg')
    aug_image = aug_transform(image)
    # aug_image_PIL = to_PIL(aug_image)
    # aug_image_PIL.save('aug.jpg')
    # import ipdb; ipdb.set_trace()
    return aug_image


class Cephalometric(data.Dataset):
    def __init__(self, pathDataset, mode="Oneshot", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None,
                 retfunc=1, use_prob=False, ret_name=False, be_consistency=False,
                 num_repeat=10):
        assert not (ref_landmark is not None and prob_map is not None), \
            f"Got both ref_landmark: \n{ref_landmark} \nand prob_map: \n{prob_map}"

        self.size = size if isinstance(size, list) else [size, size]
        self.original_size = [2400, 1935]
        self.retfunc = retfunc
        if retfunc > 0:
            print("Using new ret Function!")
            self.new_ret = True
        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')
        self.patch_size = patch_size
        self.pre_crop = pre_crop
        self.ref_landmark = ref_landmark
        self.prob_map = prob_map
        self.ret_name = ret_name
        if rand_psize:
            self.rand_psize = rand_psize
            self.patch_size = -1

        self.list = list()

        if mode in ['Oneshot', 'Train']:
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        elif mode == 'Test2':
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
        elif mode == 'Test1+2':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 400
        else:
            raise ValueError

        self.pre_trans = transforms.Compose([transforms.RandomCrop((int(2400 * 0.8), int(1935 * 0.8)))])

        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        # transforms.ColorJitter(brightness=0.15, contrast=0.25),
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        transform_list = [
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.aug_transform = transforms.Compose(transform_list)

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        num_repeat = num_repeat
        if mode == 'Train' or mode == 'Oneshot':
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)

        self.mode = mode
        self.base = 16
        self.be_consistency = be_consistency

        if use_prob:  # prob map setting
            print("Using Retfunc2() and Prob map")
            assert retfunc == 2, f" Retfunc Error, Got {retfunc}"
            if prob_map is None:
                self.prob_map = self.prob_map_from_landmarks(self.size)

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 8) * 32

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark

    def prob_map_from_landmarks(self, size=(384, 384), kernel_size=192):
        """
        Guassion Prob map from landmarks
        landmarks: [(x,y), (), ()....]
        size: (384,384)
        """
        landmarks = self.ref_landmark
        prob_maps = get_guassian_heatmaps_from_ref(landmarks=landmarks, num_classes=len(landmarks), \
                                                   image_shape=size, kernel_size=kernel_size,
                                                   sharpness=0.2)  # shape: (19, 800, 640)
        prob_map = np.sum(prob_maps, axis=0)
        prob_map = np.clip(prob_map, 0, 1)
        print("====== Save Prob map to ./imgshow")
        cv2.imwrite(f"imgshow/prob_map_ks{kernel_size}.jpg", (prob_map * 255).astype(np.uint8))
        return prob_map

    def select_point_from_prob_map(self, prob_map, size=(192, 192)):
        size_x, size_y = prob_map.shape
        assert size_x == size[0]
        assert size_y == size[1]
        chosen_x1 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_x2 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y1 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        chosen_y2 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        if prob_map[chosen_x1, chosen_y1] * np.random.random() > prob_map[chosen_x2, chosen_y2] * np.random.random():
            return chosen_x1, chosen_y1
        else:
            return chosen_x2, chosen_y2

    def retfunc1(self, index):
        """
        New Point Choosing Function without prob map
        """
        np.random.seed()
        item = self.list[index]
        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
        pad_scale = 0.05
        padding = int(pad_scale * self.size[0])
        patch_size = self.patch_size
        # raw_x, raw_y = self.select_point_from_prob_map(self.prob_map, size=self.size)
        raw_x = np.random.randint(int(pad_scale * self.size[0]), int((1 - pad_scale) * self.size[0]))
        raw_y = np.random.randint(int(pad_scale * self.size[1]), int((1 - pad_scale) * self.size[1]))

        b1_left = 0
        b1_top = 0
        b1_right = self.size[0] - patch_size
        b1_bot = self.size[1] - patch_size
        b2_left = raw_x - patch_size + 1
        b2_top = raw_y - patch_size + 1
        b2_right = raw_x
        b2_bot = raw_y
        b_left = max(b1_left, b2_left)
        b_top = max(b1_top, b2_top)
        b_right = min(b1_right, b2_right)
        b_bot = min(b1_bot, b2_bot)
        left = np.random.randint(b_left, b_right)
        top = np.random.randint(b_top, b_bot)

        margin_x = left
        margin_y = top
        cimg = item['image'][:, margin_y:margin_y + patch_size, margin_x:margin_x + patch_size]
        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_x, chosen_y = raw_x - margin_x, raw_y - margin_y

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y, chosen_x] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        chosen_y, chosen_x = temp.argmax() // patch_size, temp.argmax() % patch_size

        sift_dir = '/home1/quanquan/code/landmark/code/runs/sift/sift/s1/lm/'
        sift_response_pth = sift_dir + f"sift_responses_{item['ID']}.npy"
        sift_landmark_pth = sift_dir + f"sift_landmarks_{item['ID']}.npy"
        sift_descript_pth = sift_dir + f"sift_descriptor_{item['ID']}.npy"
        sift_response = np.load(sift_response_pth)
        sift_landmark = np.load(sift_landmark_pth)
        sift_descript = np.load(sift_descript_pth)

        sift_landmark = torch.Tensor(sift_landmark).round().long() // 2
        w, h = item['image'].shape[1] // 2, item['image'].shape[2] // 2
        # print("shape: w, h ", w, h )
        sift_descript = torch.Tensor(sift_descript)
        landmark_map = torch.zeros((w, h))
        response_map = torch.zeros((w, h))
        descript_map = torch.zeros((128, w, h))
        # descript_map[:, sift_landmark[0], sift_landmark[1]] = sift_descript
        for i, lm in enumerate(sift_landmark):
            descript_map[:, lm[0], lm[1]] = sift_descript[i]
            response_map[lm[0], lm[1]] = sift_response[i]
            landmark_map[lm[0], lm[1]] = 1

        return {'raw_imgs': item['image'],
                'crop_imgs': crop_imgs,
                'raw_loc': torch.IntTensor([raw_x, raw_y]),
                'chosen_loc': torch.IntTensor([chosen_x, chosen_y]),
                'ID': item['ID'],
                'sift_response': response_map,
                'sift_landmark': landmark_map,
                'sift_descript': descript_map}

    def __getitem__(self, index):
        if self.retfunc == 1:
            return self.retfunc1(index)
        elif self.retfunc == 3:
            raise ValueError


    def __len__(self):
        return len(self.list)



if __name__ == "__main__":
    pass