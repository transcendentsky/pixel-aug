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
# from skimage.filters.rank import entropy
from scipy.stats import entropy
from skimage.morphology import disk
from tutils import tfilename, torchvision_save
from skimage.exposure import histogram
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
    im = cv2.resize(im, (size[::-1]))
    # entr_img = entropy(im, disk(32))
    entr_img = hist_to_entropy(im_to_hist(im))
    # entr_img = cv2.resize(entr_img, (size[::-1]))
    print("entr_img ", entr_img.shape, size)
    return entr_img

AUG_PARAMS = {
    "me": { #
        "cj_brightness": [0.505, 1.997], # lm 0
        "cj_contrast": [0.340, 2.329],
    },
    "he": { # 1.62 1.83, 0.54 0.40
        "cj_brightness": [0.5617, 1.612], # lm 17
        "cj_contrast": [0.416, 1.817],
    }
}

class Cephalometric(data.Dataset):
    def __init__(self, pathDataset, mode="Train", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, use_adap_aug=False,
                 retfunc=1, use_prob=True, ret_name=False, be_consistency=False,
                 cj_brightness=0.15, cj_contrast=0.25, cj_saturation=0., cj_hue=0.,
                 sharpness=0.2, multi_output=False, hard_select=False,
                 prob_map_dir=None, entr_map_dir=None, num_repeat=10, runs_dir=None, return_entr=False):
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
        self.entr_map_dir = entr_map_dir
        self.prob_map_dir = prob_map_dir
        self.ret_name = ret_name
        self.multi_output = multi_output
        self.use_prob = use_prob
        self.hard_select = hard_select
        self.xy = np.arange(0, self.size[0] * self.size[1])
        self.runs_dir = runs_dir
        self.return_entr = return_entr
        self.use_adap_aug = use_adap_aug
        if rand_psize:
            self.rand_psize = rand_psize
            self.patch_size = -1

        self.list = list()

        if mode in ['Oneshot', 'Train']:
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == 'subtrain':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 5
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

        self.pre_trans = transforms.Compose([transforms.RandomCrop((int(2400 * 0.8), int(1935 * 0.8)))])

        self.transform = transforms.Compose([transforms.Resize(self.size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0], [1])])

        transform_list = [
            # transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.aug_transform = transforms.Compose(transform_list)
        self.aug_transform_me = transforms.Compose([
            transforms.ColorJitter(brightness=AUG_PARAMS['me']['cj_brightness'],
                                   contrast=AUG_PARAMS['me']['cj_contrast']),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])
        self.aug_transform_he = transforms.Compose([
            transforms.ColorJitter(brightness=AUG_PARAMS['he']['cj_brightness'],
                                   contrast=AUG_PARAMS['he']['cj_contrast']),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])
        self.resize_fn = transforms.Resize(self.size)

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        num_repeat = num_repeat
        if mode in ['Train', 'Oneshot', 'subtrain']:
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list += temp
        # print("DEBUG, list len", len(self.list), "num_repeat", num_repeat)
        self.mode = mode
        self.base = 16
        self.be_consistency = be_consistency
        self.prob_map_dict = None
        self.entr_value_dict = None
        self.edge_map_dict = None
        self.prob_entr_dict = None

        if use_prob:  # prob map setting
            print("Using Retfunc2() and Prob map")
            assert retfunc == 2, f" Retfunc Error, Got {retfunc}"

        self.init_mask()

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 8) * 32

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark

    def _tmp_resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[1 - i] / self.original_size[1 - i])
        return landmark

    def edge_map_from_image(self, edge_dir, inverse=False):
        print("[*] Accessing edge maps!")
        self.edge_map_dict = {}
        for i in range(150):
            name = str(i+1) + ".npy"
            path = tfilename(edge_dir, name)
            if not os.path.exists(path):
                im_path = os.path.join(self.pth_Image, "{0:03d}".format(i+1) + '.bmp')
                im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
                im = cv2.resize(im, (384, 384))
                edge_map = cv2.Canny(im, 50, 200)
                assert edge_map.shape[-2:] == (384,384), f"create entr map, Got {edge_map.shape}, it should be {(384,384)}"
                np.save(path, edge_map)
                if i == 0:
                    torchvision_save(torch.Tensor(edge_map.copy()/edge_map.max()), tfilename(self.prob_map_dir, name[:-4]+".png"))
            else:
                edge_map = np.load(path)
            if inverse:
                edge_map = edge_map.max() - edge_map
            edge_map = edge_map / edge_map.max()
            # assert entr_map.shape == (384, 384)
            self.edge_map_dict["{0:03d}".format(i+1)] = edge_map

            if i == 0:
                torchvision_save(torch.Tensor(edge_map.copy() / edge_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
        print("Got entropy maps")

    def prob_map_point(self, all_landmarks, size=(384,384)):
        # bug ??
        print("[*] Accessing prob point maps!")
        self.prob_map_dict = {}
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
            else:
                prob_map = np.load(path)
                assert prob_map.shape[-2:] == tuple(size), f"use existed probmap, Got {prob_map.shape}, it should be {size}"

            if i == 0:
                torchvision_save(torch.Tensor(prob_map.copy() / prob_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
            self.prob_map_dict["{0:03d}".format(i+1)] = prob_map

    def prob_map_for_all(self, all_landmarks, size=(384,384), sharpness=0.2, kernel_size=192, inverse=False):
        print("[*] Accessing prob maps!")
        self.prob_map_dict = {}
        for i, landmarks in enumerate(all_landmarks):
            print("Processing ", i, end="\r")
            name = str(i+1) + ".npy"
            path = tfilename(self.prob_map_dir, name)
            if not os.path.exists(path):
                prob_maps = get_guassian_heatmaps_from_ref(landmarks=landmarks, num_classes=len(landmarks), \
                                                           image_shape=size, kernel_size=kernel_size,
                                                           sharpness=sharpness)  # shape: (19, 800, 640)
                prob_map = np.sum(prob_maps, axis=0)
                prob_map = np.clip(prob_map, 0, 1)
                assert prob_map.shape[-2:] == tuple(size), f"create new probmap, Got {prob_map.shape}, it should be {size}"
                # assert prob_map.shape == (384, 384)
                np.save(path, prob_map)
            else:
                prob_map = np.load(path)
                assert prob_map.shape[-2:] == tuple(size), f"use existed probmap, Got {prob_map.shape}, it should be {size}"
            if inverse:
                prob_map = prob_map.max() - prob_map
            prob_map /= prob_map.sum()

            if i == 0:
                torchvision_save(torch.Tensor(prob_map.copy() / prob_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
            self.prob_map_dict["{0:03d}".format(i+1)] = prob_map

    def prob_map_mi(self, mi_dir):
        print("Accessing MI prob maps!")
        self.prob_map_dict = {}
        for i in range(150):
            name = str(i+1) + ".npy"
            prob_map = np.load(tfilename(mi_dir, f"{i:03d}.npy"), allow_pickle=True)
            prob_map /= prob_map.sum()
            self.prob_map_dict[f"{i+1:03d}"] = prob_map
            if i == 0:
                print("debug ", prob_map.shape)
            if i == 0:
                torchvision_save(torch.Tensor(prob_map.copy() / prob_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))

    def entr_map_from_image(self, size=(384, 384), kernel_size=192, temperature=1, inverse=False):
        print("[*] Accessing entropy maps!")
        self.entr_value_dict = {}
        for i in range(150):
            name = str(i+1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                im_path = os.path.join(self.pth_Image, "{0:03d}".format(i+1) + '.bmp')
                entr_map = get_entr_map_from_image(image_path=im_path, size=self.size)
                assert entr_map.shape[-2:] == tuple(size), f"create entr map, Got {entr_map.shape}, it should be {size}"
                np.save(path, entr_map)
            else:
                entr_map = np.load(path)
                assert entr_map.shape[-2:] == tuple(size), f"use exist entr map, Got {entr_map.shape}, it should be {size}"
            if inverse:
                entr_map = entr_map.max() - entr_map
            entr_map = entr_map ** temperature

            if i == 0:
                torchvision_save(torch.Tensor(entr_map.copy() / entr_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
                print("Writing images! ")
            self.entr_value_dict["{0:03d}".format(i+1)] = entr_map
        print("Got entropy maps")

    def entr_map_from_image_polyline(self, size=(384, 384), kernel_size=192, temperature=0.5, inverse=False):
        print("[*] Accessing entropy maps!")
        self.prob_entr_dict = {}
        for i in range(150):
            name = str(i + 1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                im_path = os.path.join(self.pth_Image, "{0:03d}".format(i + 1) + '.bmp')
                entr_map = get_entr_map_from_image(image_path=im_path, size=self.size)
                assert entr_map.shape[-2:] == tuple(size), f"create entr map, Got {entr_map.shape}, it should be {size}"
                # assert entr_map.shape == (384, 384)
                # cv2.imwrite(path[:-4] + ".jpg", entr_map, cmap='gray')
                np.save(path, entr_map)
            else:
                entr_map = np.load(path)
                assert entr_map.shape[-2:] == tuple(
                    size), f"use exist entr map, Got {entr_map.shape}, it should be {size}"
            if inverse:
                entr_map = entr_map.max() - entr_map
            # entr_map = entr_map / entr_map.max()
            entr_map = entr_map ** temperature

            if i == 0:
                torchvision_save(torch.Tensor(entr_map.copy() / entr_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
                print("Writing images! ")
            # assert entr_map.shape == (384, 384)
            self.prob_entr_dict["{0:03d}".format(i + 1)] = entr_map
        print("Got entropy maps")

    def entr_map_ushape(self, size=(384,384), temperature=0.3, inverse=False):
        print("[*] Accessing entropy maps!")
        self.prob_entr_dict = {}
        for i in range(150):
            name = str(i+1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                raise NotImplementedError
            else:
                entr_map = np.load(path)
                assert entr_map.shape[-2:] == tuple(size), f"use exist entr map, Got {entr_map.shape}, it should be {size}"
            entr_map = entr_map / entr_map.max()
            entr_map = entr_map ** temperature
            entr_map2 = (entr_map.max() - entr_map) ** temperature
            entr_map = entr_map + entr_map2

            if inverse:
                entr_map = entr_map.max() - entr_map

            if i == 0:
                torchvision_save(torch.Tensor(entr_map.copy() / entr_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
                print("Writing images! ")
            # assert entr_map.shape == (384, 384)
            self.prob_entr_dict["{0:03d}".format(i+1)] = entr_map
        print("Got entropy maps")

    def entr_map_ushape2(self, size=(384,384), temperature=0.3, inverse=False):
        print("[*] Accessing entropy maps!")
        self.prob_entr_dict = {}
        for i in range(150):
            name = str(i+1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                raise NotImplementedError
            else:
                entr_map = np.load(path)
                assert entr_map.shape[-2:] == tuple(size), f"use exist entr map, Got {entr_map.shape}, it should be {size}"
            # entr_map = 1.43-(entr_map-3.4)**2 / 32
            # entr_map = np.clip(1.07-(entr_map-3.4)**2/32, 0, 1)
            entr_map = np.clip(entr_map ** temperature, 0, 4.5**temperature)
            if inverse:
                entr_map = entr_map.max() - entr_map
            if i == 0:
                torchvision_save(torch.Tensor(entr_map.copy() / entr_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
                print("Writing images! ")
            # assert entr_map.shape == (384, 384)
            self.prob_entr_dict["{0:03d}".format(i+1)] = entr_map
        print("Got entropy maps")

    def entr_map_ushape3(self, size=(384,384), temperature=0.3, inverse=False):
        print("[*] Accessing entropy maps!")
        self.prob_entr_dict = {}
        for i in range(150):
            name = str(i+1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                raise NotImplementedError
            else:
                entr_map = np.load(path)
                assert entr_map.shape[-2:] == tuple(size), f"use exist entr map, Got {entr_map.shape}, it should be {size}"

            entr_map = entr_map / entr_map.max()
            entr_map = entr_map ** temperature
            if inverse:
                entr_map = entr_map.max() - entr_map
            if i == 0:
                torchvision_save(torch.Tensor(entr_map.copy() / entr_map.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
                print("Writing images! ")
            # assert entr_map.shape == (384, 384)
            self.prob_entr_dict["{0:03d}".format(i+1)] = entr_map
        print("Got entropy maps")

    def select_point_from_prob_entr_map(self, prob, size=(192, 192)):
        size_x, size_y = prob.shape
        assert size_x == size[0]
        assert size_y == size[1]
        chosen_x1 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y1 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        chosen_x2 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y2 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        if (prob[chosen_x1, chosen_y1]) * np.random.random() > (prob[chosen_x2, chosen_y2]) * np.random.random():
            return chosen_x1, chosen_y1
        else:
            return chosen_x2, chosen_y2

    def select_point_hard(self, prob_map, size=(192, 192)):
        prob = rearrange(prob_map, "h w -> (h w)")
        prob /= prob.sum()
        loc = np.random.choice(a=self.xy, size=1, replace=True, p=prob)[0]
        h, w = loc // self.size[1], loc % self.size[1]
        # print(_ret)
        return h, w

    def retfunc2(self, index):
        """
        New Point Choosing Function with 'PROB MAP'
        """
        # np.random.seed()
        ret_dict = {}
        item = self.list[index]
        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))

        if not self.use_prob:
            raw_h = np.random.randint(int(0.1 * self.size[0]), int(0.9 * self.size[0]))
            raw_w = np.random.randint(int(0.1 * self.size[1]), int(0.9 * self.size[1]))
        else:
            prob = self.prob_map_dict[item['ID']] if self.prob_map_dict is not None else np.ones(self.size)
            prob = prob * self.prob_entr_dict[item['ID']] if self.prob_entr_dict is not None else prob
            if self.hard_select:
                raw_h, raw_w = self.select_point_hard(prob, size=self.size)
            else:
                raw_h, raw_w = self.select_point_from_prob_entr_map(prob, size=self.size)
        if self.return_entr:
            entr_value = self.entr_value_dict[item['ID']][raw_h, raw_w]
            ret_dict['entr_value'] = entr_value

        crop_imgs, chosen_h, chosen_w, corner = self._get_patch(item, raw_h, raw_w)
        if self.multi_output:
            crop_imgs2, chosen_h2, chosen_w2, _ = self._get_patch(item, raw_h, raw_w, corner=corner)
            return {**ret_dict, **{'raw_imgs': item['image'], 'raw_loc': torch.LongTensor([raw_h, raw_w]), 'ID': item['ID'],
                    'crop_imgs1': crop_imgs , 'chosen_loc1': torch.LongTensor([chosen_h , chosen_w ]),
                    'crop_imgs2': crop_imgs2, 'chosen_loc2': torch.LongTensor([chosen_h2, chosen_w2]),}}
        else:
            return {**ret_dict, **{'raw_imgs': item['image'], 'raw_loc': torch.LongTensor([raw_h, raw_w]), 'ID': item['ID'],
                    'crop_imgs': crop_imgs , 'chosen_loc': torch.LongTensor([chosen_h, chosen_w]),}}

    def retfun3_debug(self, index):
        ret_dict = {}
        item = self.list[index]
        pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
        ori_img = Image.open(pth_img).convert('RGB')
        raw_h = np.random.randint(int(0.1 * self.size[0]), int(0.9 * self.size[0]))
        raw_w = np.random.randint(int(0.1 * self.size[1]), int(0.9 * self.size[1]))
        img1 = self.transform(ori_img)
        img2 = self.aug_transform(self.resize_fn(ori_img))
        entr_value = self.entr_value_dict[item['ID']][raw_h, raw_w]
        ret_dict['entr_value'] = entr_value
        landmarks = self.ref_landmarks(index)
        return {"img1": img1, "img2": img2, "raw_loc": torch.LongTensor([raw_h, raw_w]), "entr_value": entr_value, "entr_map": self.entr_value_dict[item['ID']], "landmarks": landmarks}

    def ref_landmarks(self, index):
        np.random.seed()
        item = self.list[index]
        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID'] + '.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID'] + '.txt')) as f2:
                for i in range(19):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5 * (int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self._tmp_resize_landmark(landmark))
        return landmark_list

    def _get_patch(self, item, raw_h, raw_w, corner=None):
        patch_size = self.patch_size
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
        if corner is None:
            if b_left > b_right or b_top > b_bot:
                print("1  ", b_left, b_right, b_top, b_bot)
                print("2  ", raw_h, raw_w, self.size)
                print("3  ", b1_left, b1_right, b1_top, b1_bot)
                print("4  ", b2_left, b2_right, b2_top, b2_bot)
                import ipdb; ipdb.set_trace()
            left = np.random.randint(b_left, b_right) if b_right > b_left else b_left
            top = np.random.randint(b_top, b_bot) if b_bot > b_top else b_top
        else:
            left, top = corner

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

    def augment_patch_adap(self, tensor, entr_value):
        image = to_PIL(tensor)
        if entr_value < 4.5 and entr_value > 3.5:
            aug_image = self.aug_transform_me(image)
        else:
            aug_image = self.aug_transform_he(image)
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
        elif self.retfunc == 3:
            return self.retfun3_debug(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.list)

