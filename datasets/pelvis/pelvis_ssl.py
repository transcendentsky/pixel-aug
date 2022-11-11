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
from tutils import tfilename


def to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8).transpose(1, 2, 0))
    return images


def augment_patch(tensor, aug_transform):
    image = to_PIL(tensor)
    aug_image = aug_transform(image)
    return aug_image


class Pelvis(data.Dataset):
    def __init__(self, pathDataset, mode="Train", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None,
                 retfunc=1, use_prob=False, ret_name=False, be_consistency=False,
                 num_repeat=10):
        assert not (ref_landmark is not None and prob_map is not None), \
            f"Got both ref_landmark: \n{ref_landmark} \nand prob_map: \n{prob_map}"

        self.path = pathDataset
        self.size = size if isinstance(size, list) else [size, size]
        self.original_size = [2400, 1935]
        self.retfunc = retfunc

        self.patch_size = patch_size
        self.pre_crop = pre_crop
        self.ref_landmark = ref_landmark
        self.prob_map = prob_map
        self.ret_name = ret_name
        self.init_dataset()

        self.pre_trans = transforms.Compose([transforms.RandomCrop((int(2400 * 0.8), int(1935 * 0.8)))])
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            # transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1]),
        ])
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])])

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

    def retfunc1(self, index):
        np.random.seed()
        img, landmarks, spacing = self.get_data(index)
        return self._process1(index, img)

    def _process1(self, index, img):
        """
        New Point Choosing Function without prob map
        """
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

        return {'raw_imgs': img, 'crop_imgs': crop_imgs, 'raw_loc': torch.LongTensor([raw_h, raw_w]), 'chosen_loc': torch.LongTensor([chosen_h, chosen_w]), 'ID': index }

    def retfunc2(self, index):
        """
        New Point Choosing Function with 'PROB MAP'
        """
        np.random.seed()
        item = self.list[index]
        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
            # print("[debug] ", item['image'].shape, item['extra_image'].shape)

        # patch_size = int(self.patch_scale * self.size[0])
        padding = int(0.1 * self.size[0])
        patch_size = self.patch_size
        # prob_map = self.prob_map_from_landmarks(self.size)
        raw_w, raw_h = self.select_point_from_prob_map(self.prob_map, size=self.size)

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
        # print("margin x y", margin_h, margin_w, patch_size)
        # import ipdb; ipdb.set_trace()
        cimg = item['image'][:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size]
        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_w, chosen_h = raw_w - margin_w, raw_h - margin_h

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_h, chosen_w] = 1
        # print("[debug2s] ", crop_imgs.shape, temp.shape)
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        # to_PIL(crop_imgs).save('img_after.jpg')
        temp = temp[3]
        # print(chosen_h, chosen_w)
        chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size
        # print(chosen_h, chosen_w)
        # import ipdb; ipdb.set_trace()
        # return item['image'], crop_imgs, chosen_h, chosen_w, raw_h, raw_w
        return {'raw_imgs': item['image'], 'crop_imgs': crop_imgs, 'raw_loc': torch.LongTensor([raw_h, raw_w]), 'chosen_loc': torch.LongTensor([chosen_h, chosen_w]), 'ID': item['ID']}

    def __getitem__(self, index):
        if self.retfunc == 1:
            return self.retfunc1(index)
        elif self.retfunc == 2:
            return self.retfunc2(index)
        else:
            raise ValueError

