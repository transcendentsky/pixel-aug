"""
    ceph_ssl_adap + multi patches output
"""
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
# from scipy.stats import entropy
from skimage.morphology import disk
from tutils import tfilename, torchvision_save
from skimage.exposure import histogram
import csv
from einops import rearrange


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


def im_to_hist(im, patch_size=64, temp=None):
    assert im.shape == (384, 384), f"Got {im.shape}"
    assert len(im.shape) == 2, f"GOt {im.shape}"
    h, w = im.shape
    # mi = np.zeros((h, w))
    ps_h = patch_size // 2
    fea_matrix = np.zeros((256, h, w))
    for i in range(h):
        for j in range(w):
            l1 = max(0, i-ps_h)
            l2 = max(0, j-ps_h)
            patch = im[l1:i+ps_h, l2:j+ps_h]
            hist, idx = histogram(patch, nbins=256)
            # fea = np.zeros((256,))
            for hi, idi in zip(hist, idx):
                # print(hi, idi, i, j)
                fea_matrix[idi, i, j] = hi
            # fea_matrix[:, i, j] /= fea_matrix[:, i, j].sum()
            # assert
            # entr = entropy(fea_matrix[:, i, j])
            # if np.isnan(entr) :
            #     import ipdb;ipdb.set_trace()
            # mii = mutual_info_score(temp, fea)
            # mi[i, j] = mii
    return fea_matrix


def hist_to_entropy(hist):
    c, h, w = hist.shape
    entr_map = np.zeros((h, w))
    for hi in range(h):
        for wi in range(w):
            fea = hist[:, hi, wi]
            entr = entropy(fea)
            if np.isnan(entr) :
                import ipdb; ipdb.set_trace()
            entr_map[hi, wi] = entr
    return entr_map


def get_entr_map_from_image(image_path, size):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    parent, name = os.path.split(image_path)
    # im = cv2.resize(im, (size[::-1]))
    entr_img = entropy(im, disk(10))
    # entr_img = hist_to_entropy(im_to_hist(im))
    entr_img = cv2.resize(entr_img, (size[::-1]))
    print("entr_img ", name, entr_img.shape, size)
    torchvision_save(torch.Tensor(entr_img), tfilename("./cache/hand_entr/"+name))
    return entr_img


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


# '/home/quanquan/hand/hand/histo-norm/'
class HandXray(data.Dataset):
    def __init__(self, pathDataset='/home/quanquan/hand/hand/',\
                 mode="Train", size=[384, 384], extra_aug_loss=False, patch_size=192, rand_psize=False,
                 retfunc=1, use_prob=True, multi_output=True, cj_brightness=0.15, cj_contrast=0.25,
                 prob_map_dir=None, entr_map_dir=None, num_repeat=1, prob_mode="gaussian"):
        """
        extra_aug_loss: return an additional image for image augmentation consistence loss
        """
        self.size = size
        self.extra_aug_loss = extra_aug_loss  # consistency loss
        self.pth_Image = tfilename(pathDataset, "jpg")
        self.patch_size = patch_size
        self.use_prob = use_prob
        self.multi_output = multi_output
        # self.prob_map = prob_map
        self.entr_map_dir = entr_map_dir
        self.prob_map_dir = prob_map_dir
        self.retfunc = retfunc
        self.mode = mode
        self.base = 16
        self.num_repeat = num_repeat
        self.prob_mode = prob_mode

        self.list = [x.path for x in os.scandir(self.pth_Image) if x.name.endswith(".jpg")]
        self.list.sort()
        # print(self.list)
        label_path = tfilename(pathDataset, "all.csv")
        self.landmarks = get_csv_content(label_path)
        self.test_list = self.list[:300]
        self.landmarks_test = self.landmarks[:300]
        self.train_list = self.list[300:]
        self.landmarks_train = self.landmarks[300:]
        self.xy = np.arange(0, self.size[0] * self.size[1])

        if mode in ["Oneshot", "Train"]:
            self.istrain = True
        elif mode in ["Test1", "Test"]:
            self.istrain = False
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.], [1.]),
        ])

        self.transform_resize = transforms.Resize(self.size)
        self.transform_tensor = transforms.ToTensor()

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])

        if mode in ['Train', 'Oneshot']:
            self.loading_list = self.train_list
            self.real_len = len(self.loading_list)
        else:
            raise NotImplementedError

    def prob_map_for_all(self, all_landmarks, size=(384,384), sharpness=0.2, kernel_size=192):
        print("Accessing prob maps!")
        self.prob_map_dict = []
        for i, landmarks in enumerate(all_landmarks):
            name = str(i+1) + ".npy"
            path = tfilename(self.prob_map_dir, name)
            print("Processing ", path, end="\r")
            if not os.path.exists(path):
                prob_maps = get_guassian_heatmaps_from_ref(landmarks=landmarks, num_classes=len(landmarks), \
                                                           image_shape=size, kernel_size=kernel_size,
                                                           sharpness=sharpness)  # shape: (19, 800, 640)
                prob_map = np.sum(prob_maps, axis=0)
                prob_map = np.clip(prob_map, 0, 1)
                assert prob_map.shape[-2:] == tuple(size), f"create new probmap, Got {prob_map.shape}, it should be {size}"
                assert prob_map.max() > 0, f"Got {prob_map.max()}"
                prob_map = prob_map / prob_map.sum()
                # assert prob_map.shape == (384, 384)
                np.save(path, prob_map)
                if i == 0:
                    torchvision_save(torch.Tensor(prob_map.copy()/prob_map.max()), tfilename(self.prob_map_dir, name[:-4]+".png"))
            else:
                prob_map = np.load(path)
                assert prob_map.shape[-2:] == tuple(size), f"use existed probmap, Got {prob_map.shape}, it should be {size}"
            self.prob_map_dict.append(prob_map)

    def prob_map_point(self, all_landmarks, size=(384,384)):
        # bug ??
        print("Accessing prob maps!")
        self.prob_map_dict = []
        for i, landmarks in enumerate(all_landmarks):
            name = str(i+1) + ".npy"
            path = tfilename(self.prob_map_dir, name)
            print("Processing ", path, end="\r")
            if not os.path.exists(path):
                prob_map = np.zeros(size)
                for lm in landmarks:
                    prob_map[lm[1], lm[0]] = 1
                assert prob_map.shape[-2:] == tuple(size), f"create new probmap, Got {prob_map.shape}, it should be {size}"
                assert prob_map.max() > 0, f"Got {prob_map.max()}"
                prob_map = prob_map / prob_map.sum()
                np.save(path, prob_map)
                if i == 0:
                    torchvision_save(torch.Tensor(prob_map.copy()/prob_map.max()), tfilename(self.prob_map_dir, name[:-4]+".png"))
            else:
                prob_map = np.load(path)
                assert prob_map.shape[-2:] == tuple(size), f"use existed probmap, Got {prob_map.shape}, it should be {size}"
            self.prob_map_dict.append(prob_map)

    def entr_map_from_image(self, size=(384, 384), temperature=0.1, inverse=False):
        print("Accessing entropy maps!")
        self.entr_map_dict = []
        for i, im_path in enumerate(self.train_list):
            print("Processing ", im_path, end="\r")
            name = str(i+1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                entr_map = get_entr_map_from_image(image_path=im_path, size=self.size)
                assert entr_map.shape[-2:] == tuple(size), f"create entr map, Got {entr_map.shape}, it should be {size}"
                np.save(path, entr_map)
            else:
                entr_map = np.load(path, allow_pickle=True) # allow_pickle=True for debug
                assert entr_map.shape[-2:] == tuple(size), f"use exist entr map, Got {entr_map.shape}, it should be {size}"
            if inverse:
                entr_map = entr_map.max() - entr_map
            entr_map = entr_map ** temperature
            entr_map = entr_map / entr_map.sum()
            # assert entr_map.shape == (384, 384)
            # self.entr_map_dict["{0:03d}".format(i+1)] = entr_map
            if i == 0:
                torchvision_save(torch.Tensor(entr_map.copy() / entr_map.max()),
                                 tfilename("./tmp/hand_ssl_adapm/", name[:-4] + ".png"))
            self.entr_map_dict.append(entr_map)
        print("Got entropy maps")

    def save_ref_landmarks(self, all_landmarks):
        self.ref_landmarks = all_landmarks

    def select_point_from_prob_map(self, prob_map, size=(192, 192)):
        prob = rearrange(prob_map, "h w -> (h w)")
        loc = np.random.choice(a=self.xy, size=1, replace=True, p=prob)[0]
        h, w = loc // self.size[1], loc % self.size[1]
        # print(_ret)
        # if h < 0 or w < 0:
        #     print(h, w)
        #     import ipdb; ipdb.set_trace()
        return h, w

    def select_point_from_point(self, landmarks, size=(192, 192)):
        idx = np.random.choice(a=np.arange(0, len(landmarks)), size=1, replace=True)[0]
        h, w = landmarks[idx][1], landmarks[idx][0]
        return h, w

    def retfunc1(self, index):
        index = index % self.real_len
        np.random.seed()
        img, landmarks, spacing = self._get_data(index % len(self.train_list))
        return self._process1(index % len(self.train_list), img)

    def _get_data(self, index):
        index = index % self.real_len
        img_pil = Image.open(self.train_list[index]).convert('RGB')
        landmarks = None
        spacing = None
        return img_pil, landmarks, spacing

    def _process1(self, index, img):
        """
        New Point Choosing Function without prob map
        """
        img = self.transform(img)
        patch_size = self.patch_size
        if not self.use_prob:
            raw_h = np.random.randint(int(0.05 * self.size[0]), int(0.95 * self.size[0]))
            raw_w = np.random.randint(int(0.05 * self.size[1]), int(0.95 * self.size[1]))
        else:
            if self.prob_mode == "gaussian":
                raw_h, raw_w = self.select_point_from_prob_map(self.prob_map_dict[index], size=self.size)
            elif self.prob_mode == "point":
                raw_h, raw_w = self.select_point_from_point(self.ref_landmarks[index], size=self.size)
            elif self.prob_mode == "entr":
                raw_h, raw_w = self.select_point_from_prob_map(self.entr_map_dict[index], size=self.size)
            else: raise NotImplementedError
            # raw_h, raw_w = self.select_point_from_entr_map(self.entr_map_dict[index], size=self.size)

        crop_imgs, chosen_h, chosen_w, corner = self._get_patch(img, raw_h, raw_w)
        if self.multi_output:
            corner = corner if self.multi_output else None
            crop_imgs2, chosen_h2, chosen_w2, _ = self._get_patch(img, raw_h, raw_w, corner=corner)

            return {'raw_imgs': img, 'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'crop_imgs1': crop_imgs, 'chosen_loc1': torch.LongTensor([chosen_h, chosen_w]),
                'crop_imgs2': crop_imgs2, 'chosen_loc2': torch.LongTensor([chosen_h2, chosen_w2]), 'ID': index}
        else:
            return {'raw_imgs': img, 'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'crop_imgs': crop_imgs, 'chosen_loc': torch.LongTensor([chosen_h, chosen_w]), 'ID': index}

    def _get_patch(self, img, raw_h, raw_w, corner=None):
        patch_size = self.patch_size
        if corner is None:
            b1_left = 0
            b1_top = 0
            b1_right = self.size[1] - patch_size
            b1_bot = self.size[0] - patch_size
            b2_left = raw_w - patch_size + 1
            b2_top = raw_h - patch_size + 1
            b2_right = raw_w
            b2_bot = raw_h
            b_left = max(b1_left, b2_left)
            b_top = max(b1_top, b2_top)
            b_right = min(b1_right, b2_right)
            b_bot = min(b1_bot, b2_bot)
            if b_left > b_right or b_top > b_bot:
                print("1  ", b_left, b_right, b_top, b_bot)
                print("2  ", raw_h, raw_w, self.size)
                print("3  ", b1_left, b1_right, b1_top, b1_bot)
                print("4  ", b2_left, b2_right, b2_top, b2_bot)
                raise ValueError
            left = np.random.randint(b_left, b_right) if b_right > b_left else b_left
            top = np.random.randint(b_top, b_bot) if b_bot > b_top else b_top
        else:
            left, top = corner

        margin_w = left
        margin_h = top
        # print("margin x y", margin_h, margin_w, patch_size)
        # import ipdb; ipdb.set_trace()
        cimg = img[:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size]
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
        # chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size
        chosen_h = torch.div(temp.argmax(), patch_size, rounding_mode="trunc")
        chosen_w = temp.argmax() % patch_size
        return crop_imgs, chosen_h, chosen_w, (left, top)


    def __getitem__(self, index):
        if self.retfunc == 1:
            return self.retfunc1(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.loading_list) * self.num_repeat

