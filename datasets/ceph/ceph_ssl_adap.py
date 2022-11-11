"""
    from cas-qs/datasets/data_loader_gp.py
    debug 1: cos_map * entr map
    debug 2: cos_map + entr map
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
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tutils import tfilename, torchvision_save
from einops import rearrange


def to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8).transpose(1, 2, 0))
    return images


def augment_patch(tensor, aug_transform):
    image = to_PIL(tensor)
    aug_image = aug_transform(image)
    return aug_image


def get_entr_map_from_image(image_path):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    entr_img = entropy(im, disk(10))
    entr_img = cv2.resize(entr_img, (384,384))
    return entr_img

# (im) mean : 0.5725684210526307 1.655463157894736 0.4304421052631574 1.8743017543859652
# low: 0.5428499999999994 1.8109000000000002 0.3904666666666666 2.081200000000001
# high: 0.5941818181818173 1.5424181818181808 0.4595151515151508 1.7238303030303037

# (patch) mean: 0.5380842105263149 1.7744526315789468 0.38446315789473645 2.032954385964913
# low : 0.5054749999999995 1.9973750000000001 0.34063333333333323 2.3298333333333345
# high: 0.5617999999999989 1.612327272727272 0.4163393939393933 1.8170424242424248
AUG_PARAMS = {
    "me": { # # lm 0
        "cj_brightness": [0.505, 1.997],
        "cj_contrast": [0.340, 2.329],
    },
    "he": { # 1.62 1.83, 0.54 0.40 # lm 17
        "cj_brightness": [0.5617, 1.612],
        "cj_contrast": [0.416, 1.817],
    }
}


class Cephalometric(data.Dataset):
    def __init__(self, pathDataset, mode="Train", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None,
                 retfunc=1, use_prob=True, ret_name=False, be_consistency=False,
                 cj_brightness=0.15, cj_contrast=0.25, cj_saturation=0., cj_hue=0.,
                 sharpness=0.2, hard_select=True, use_adap_aug=False,
                 prob_map_dir=None, entr_map_dir=None, num_repeat=10, runs_dir=None, 
                 adap_params=None):
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
        self.entr_map_dir = entr_map_dir
        self.prob_map_dir = prob_map_dir
        self.ret_name = ret_name
        self.hard_select = hard_select
        self.use_adap_aug = use_adap_aug
        self.runs_dir = runs_dir
        self.adap_params = adap_params
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
        elif mode == 'subtest':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 171
        else:
            raise ValueError

        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])
        self.aug_transform_le = transforms.Compose([
            transforms.ColorJitter(brightness=self.adap_params['me']['cj_brightness'], contrast=self.adap_params['me']['cj_contrast']),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])
        self.aug_transform_me = transforms.Compose([
            transforms.ColorJitter(brightness=self.adap_params['me']['cj_brightness'], contrast=self.adap_params['me']['cj_contrast']),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])
        self.aug_transform_he = transforms.Compose([
            transforms.ColorJitter(brightness=self.adap_params['he']['cj_brightness'], contrast=self.adap_params['he']['cj_contrast']),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])
        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        num_repeat = num_repeat
        if mode in ['Train', 'Oneshot']:
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)

        self.mode = mode
        self.base = 16
        self.be_consistency = be_consistency

        if use_prob:  # prob map setting
            print("Using Retfunc2() and Prob map")
            assert retfunc == 2, f" Retfunc Error, Got {retfunc}"
            self.xy = np.arange(0, self.size[0] * self.size[1])

        self.init_mask()
        self.prob_entr_dict = None
        self.entr_value_dict = None
        self.prob_map_dict = None

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 8) * 32

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark

    def prob_map_for_all(self, all_landmarks, size=(384,384), sharpness=0.2):
        print("Get prob maps!")
        self.prob_map_dict = {}
        for i, landmarks in enumerate(all_landmarks):
            name = str(i+1) + ".npy"
            path = tfilename(self.prob_map_dir, name)
            if not os.path.exists(path):
                prob_maps = get_guassian_heatmaps_from_ref(landmarks=landmarks, num_classes=len(landmarks), \
                                                           image_shape=size, kernel_size=192,
                                                           sharpness=sharpness)  # shape: (19, 800, 640)
                prob_map = np.sum(prob_maps, axis=0)
                prob_map = np.clip(prob_map, 0, 1)
                assert prob_map.shape == (384, 384)
                np.save(path, prob_map)
            else:
                prob_map = np.load(path)
            self.prob_map_dict["{0:03d}".format(i+1)] = prob_map

    def entr_map_from_image(self, size=(384, 384), temperature=1):
        print("Get entropy maps!")
        self.entr_value_dict = {}
        for i in range(150):
            name = str(i+1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                im_path = os.path.join(self.pth_Image, "{0:03d}".format(i+1) + '.bmp')
                entr_map = get_entr_map_from_image(image_path=im_path)
                assert entr_map.shape == (384, 384)
                np.save(path, entr_map)

            else:
                entr_map = np.load(path)
                entr_map = entr_map ** 1
            assert entr_map.shape == (384, 384)
            self.entr_value_dict["{0:03d}".format(i+1)] = entr_map

    def entr_map_from_image3(self, thresholds=[0,999], inverse=False, temperature=0.5):
        """ adjusted entr map, in-threshold: 1, out-threshold: 0 """
        print("Get entropy maps! 222 ")
        self.prob_entr_dict = {}
        ratio_list = []
        for i in range(150):
            name = str(i+1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                im_path = os.path.join(self.pth_Image, "{0:03d}".format(i+1) + '.bmp')
                entr_map = get_entr_map_from_image(image_path=im_path)
                assert entr_map.shape == (384, 384)
                # cv2.imwrite(path[:-4] + ".jpg", entr_map, cmap='gray')
                np.save(path, entr_map)
            else:
                entr_map = np.load(path)
            # print()
            high_entr = np.ones_like(entr_map)
            high_entr[np.where(entr_map<thresholds[0])] = 0
            high_entr[np.where(entr_map>thresholds[1])] = 0
            assert entr_map.shape == (384, 384)
            if inverse:
                high_entr = 1 - high_entr
            ratio_list.append(high_entr.sum() / (384*384))
            self.prob_entr_dict["{0:03d}".format(i+1)] = high_entr
            if i == 0:
                torchvision_save(torch.Tensor(high_entr.copy() / high_entr.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
        print("avg ratio: ", np.array(ratio_list).mean())
        return np.array(ratio_list).mean()

    def entr_map_ushape(self, size=(384, 384), temperature=0.3):
        print("Get entropy maps!")
        self.prob_entr_dict = {}
        for i in range(150):
            name = str(i + 1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            entr_map = np.load(path)
            entr_map = entr_map ** temperature
            entr_map = np.clip(entr_map ** temperature, 0, 4.5**temperature)
            assert entr_map.shape == (384, 384)
            self.prob_entr_dict["{0:03d}".format(i + 1)] = entr_map

    def select_point_from_prob_entr_map(self, prob_map, entr_map, size=(192, 192)):
        size_x, size_y = prob_map.shape
        assert size_x == size[0]
        assert size_y == size[1]
        chosen_x1 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y1 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        chosen_x2 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y2 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        chosen_x3 = np.random.randint(int(0.05 * size_x), int(0.95 * size_x))
        chosen_y3 = np.random.randint(int(0.05 * size_y), int(0.95 * size_y))
        if (prob_map[chosen_x1, chosen_y1] * entr_map[chosen_x1, chosen_y1]) * np.random.random() > (prob_map[chosen_x2, chosen_y2] * entr_map[chosen_x2, chosen_y2]) * np.random.random():
            return chosen_x1, chosen_y1
        else:
            if (prob_map[chosen_x2, chosen_y2] * entr_map[chosen_x2, chosen_y2]) < (prob_map[chosen_x3, chosen_y3] * entr_map[chosen_x3, chosen_y3]):
                return chosen_x3, chosen_y3
            return chosen_x2, chosen_y2

    def select_point_from_prob_map_hard(self, prob_map, entr_map, size=(192, 192)):
        """ follow probmap firmly """
        prob = prob_map * entr_map
        prob = rearrange(prob, "h w -> (h w)")
        loc = np.random.choice(a=self.xy, size=1, replace=False, p=prob)
        return loc // self.size[1], loc % self.size[1]
        
    def select_point_from_prob_map_hard2(self, prob_map, size=(384,384), **kwargs):
        prob = rearrange(prob_map, "h w -> (h w)")
        assert prob.sum() >0, f"Got {prob.sum()}, name {kwargs['name']}"
        prob = prob / prob.sum()
        loc = np.random.choice(a=self.xy, size=1, replace=True, p=prob)[0]
        return loc // self.size[1], loc % self.size[1]

    def retfunc2(self, index, hard=False):
        """
        New Point Choosing Function with 'PROB MAP'
        """
        np.random.seed()
        item = self.list[index]
        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))

        padding = int(0.1 * self.size[0])
        patch_size = self.patch_size
        if self.hard_select:
            # raw_h, raw_w = self.select_point_from_prob_map_hard(self.prob_map_dict[item['ID']], self.prob_entr_dict[item['ID']], size=self.size)
            raw_h, raw_w = self.select_point_from_prob_map_hard2(self.prob_entr_dict[item['ID']], size=self.size, name=item['ID'])
        else:
            raw_h, raw_w = self.select_point_from_prob_entr_map(self.prob_map_dict[item['ID']], self.prob_entr_dict[item['ID']], size=self.size)

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
        # left = np.random.randint(b_left, b_right)
        # top = np.random.randint(b_top, b_bot)
        assert b_left <= b_right
        assert b_top <= b_bot
        left = np.random.randint(b_left, b_right) if b_left < b_right else b_left
        top = np.random.randint(b_top, b_bot) if b_top < b_bot else b_top

        margin_w = left
        margin_h = top
        # print("margin x y", margin_h, margin_w, patch_size)
        # import ipdb; ipdb.set_trace()
        cimg = item['image'][:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size]

        if self.use_adap_aug:
            crop_imgs = self.augment_patch_adap(cimg, self.entr_value_dict[item['ID']][raw_h, raw_w])
        else:
            crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_w, chosen_h = raw_w - margin_w, raw_h - margin_h

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_h, chosen_w] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        chosen_h = torch.div(temp.argmax(), patch_size, rounding_mode="trunc")
        chosen_w = temp.argmax() % patch_size
        return {'raw_imgs': item['image'], 'crop_imgs': crop_imgs, 'raw_loc': torch.LongTensor([raw_h, raw_w]), 'chosen_loc': torch.LongTensor([chosen_h, chosen_w]), 'ID': item['ID']}

    def augment_patch_adap(self, tensor, entr_value):
        image = to_PIL(tensor)
        if entr_value < 2:
            aug_image = self.aug_transform_le(image)
        elif entr_value < 4.5 and entr_value >=2:
            aug_image = self.aug_transform_me(image)
        elif entr_value >= 4.5:
            aug_image = self.aug_transform_he(image)
        else:
            raise NotImplementedError
        return aug_image

    def init_mask(self):
        # gen mask
        self.Radius = int(max(self.size) * 0.05)
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1

        # gen offset
        self.mask = mask
        self.guassian_mask = guassian_mask

        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

    def gen_mask(self, loc, img_shape=[384, 384]):
        landmark = loc
        y, x = self.size[0], self.size[1]
        mask = torch.zeros((y, x), dtype=torch.float)
        offset_x = torch.zeros((y, x), dtype=torch.float)
        offset_y = torch.zeros((y, x), dtype=torch.float)
        margin_x_left = max(0, landmark[0] - self.Radius)
        margin_x_right = min(x, landmark[0] + self.Radius)
        margin_y_bottom = max(0, landmark[1] - self.Radius)
        margin_y_top = min(y, landmark[1] + self.Radius)
        mask[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        offset_x[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        offset_y[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        return mask, offset_y, offset_x

    def __getitem__(self, index):
        if self.retfunc == 1:
            # return self.retfunc1(index)
            raise NotImplemented
        elif self.retfunc == 2:
            return self.retfunc2(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.list)

