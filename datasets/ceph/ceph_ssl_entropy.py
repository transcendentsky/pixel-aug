"""
    Copied from ceph_ssl_adap.py
    only entr map
    Use 0-1 entr map for ablation study
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
from scipy.stats import entropy as entropy2
from skimage.morphology import disk
from tutils import tfilename, torchvision_save
from skimage.exposure import histogram
from einops import rearrange
# from rich.progress import Progress, BarColumn, TextColumn
# from rich.table import Column
from tqdm import tqdm

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
            # entr = entropy2(fea_matrix[:, i, j])
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
            # print(fea.shape)
            entr = entropy2(fea)
            if np.isnan(entr) :
                import ipdb; ipdb.set_trace()
            entr_map[hi, wi] = entr
    return entr_map


def get_entr_map_from_image(image_path, size=(384,384)):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert os.path.exists(image_path), f"Got error path, {image_path}"
    # im = cv2.resize(im, (size[::-1]))
    entr_img = entropy(im, disk(128))
    # entr_img = hist_to_entropy(im_to_hist(im))
    entr_img = cv2.resize(entr_img, (size[::-1]))
    # print("entr_img ", entr_img.shape, size)
    return entr_img


class Cephalometric(data.Dataset):
    def __init__(self, pathDataset, mode="Train", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None,
                 retfunc=1, use_prob=True, ret_name=False, be_consistency=False,
                 cj_brightness=0.15, cj_contrast=0.25, cj_saturation=0., cj_hue=0.,
                 sharpness=0.2, hard_select=False, entr_temp=1,
                 prob_map_dir=None, entr_map_dir=None, num_repeat=10, runs_dir=None):
        assert not (ref_landmark is not None and prob_map is not None), \
            f"Got both ref_landmark: \n{ref_landmark} \nand prob_map: \n{prob_map}"

        self.runs_dir = runs_dir
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
        self.entr_map_dict = None
        self.ret_name = ret_name
        self.mode = mode
        self.base = 16
        self.be_consistency = be_consistency
        self.hard_select = hard_select
        self.entr_temp = entr_temp
        if rand_psize:
            self.rand_psize = rand_psize
            self.patch_size = -1

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

        self.transform = transforms.Compose([transforms.Resize(self.size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0], [1])])
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])])
        # for high entropy areas
        self.aug_transform_he = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])])
        # for medium entropy areas
        self.aug_transform_me = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])])

        self.list = list()
        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})
        num_repeat = num_repeat
        if mode in ['Train', 'Oneshot']:
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)

        self.xy = np.arange(0, self.size[0] * self.size[1])
        self.init_mask()

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 8) * 32

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark

    def entr_map_from_image_old(self, temperature=1, inverse=False):
        """ 0-1 entr map """
        print("Get entropy maps!")
        self.entr_map_dict = {}
        for i in range(150):
            name = str(i + 1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                im_path = os.path.join(self.pth_Image, "{0:03d}".format(i + 1) + '.bmp')
                entr_map = get_entr_map_from_image(image_path=im_path)
                assert entr_map.shape == (384, 384)
                # cv2.imwrite(path[:-4] + ".jpg", entr_map, cmap='gray')
                np.save(path, entr_map)
            else:
                entr_map = np.load(path)
            entr_map = (entr_map.max() - entr_map) if inverse else entr_map
            entr_map = entr_map ** temperature
            # entr_map /= entr_map.sum()
            self.entr_map_dict["{0:03d}".format(i + 1)] = entr_map
            if i == 0:
                cv2.imwrite(tfilename(f"./tmp/bi_select_t{temperature}.jpg"), (entr_map/entr_map.max() * 255).astype(int))
                print("==========================================")

    def entr_map_from_image(self, size=(384, 384), kernel_size=192, threshold=999.0, threshold2=-1, inverse=False, temperature=1, ratio=None):
        """ 0-1 entr map """
        print("Get entropy maps!")
        assert threshold2 < threshold
        self.entr_map_dict = {}
        select_per_list = []
        entr_per_list = []
        for i in tqdm(range(150)):
            # progress.print(i, end="\r")
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
            entr_map = entr_map ** temperature
            # print()
            bi_entr = np.zeros_like(entr_map)
            bi_entr[np.where(entr_map>threshold)] = 1
            ###
            bi_noise = np.zeros_like(entr_map)
            bi_noise[np.where(entr_map<threshold2)] = 1
            assert entr_map.shape == (384, 384)
            bi_select = bi_entr + bi_noise
            entr_per = bi_entr.sum() / bi_select.sum()
            select_per = bi_select.sum() / (384*384)
            entr_per_list.append(entr_per)
            select_per_list.append(select_per)
            self.entr_map_dict["{0:03d}".format(i+1)] = bi_entr
            if i == 0 and self.runs_dir is not None:
                cv2.imwrite(tfilename(self.runs_dir, f"bi_select_t{temperature}.jpg"), (bi_select*255).astype(int))
                print("================= imwrite img ======================")
        print("mean select percentage: ", np.array(select_per_list).mean())
        self.ratio = ratio
        return np.array(entr_per_list).mean(), np.array(select_per_list).mean()
        # import ipdb; ipdb.set_trace()

    def entr_map_from_image2(self, threshold=3.0, inverse=False, temperature=0.5, weight=0.5):
        """ adjusted entr map """
        print("Get entropy maps! 222 ")
        self.entr_map_dict = {}
        ratio_list = []
        rratio_list = []
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
            high_entr = np.zeros_like(entr_map)
            high_entr[np.where(entr_map>threshold)] = 1
            assert entr_map.shape == (384, 384)
            entr_map = entr_map ** temperature

            rratio = (high_entr * entr_map).sum() / entr_map.sum()
            print("Adjusting porb map, high entr ratio: ", high_entr.sum() / (384*384), rratio)
            ratio_list.append(high_entr.sum() / (384*384))
            rratio_list.append(rratio)
            entr1 = (high_entr * entr_map) / (high_entr * entr_map).sum() * weight
            entr2 = ((1-high_entr) * entr_map) / ((1-high_entr) * entr_map).sum() * (1-weight)
            entr_map2 = (entr1 + entr2)
            entr_map2 = entr_map2 / entr_map2.sum()
            # assert entr_map2.sum() == 1, f"Got {entr_map2.shape, entr_map2.sum()}"
            self.entr_map_dict["{0:03d}".format(i+1)] = entr_map2
            if i == 0:
                cv2.imwrite(tfilename(f"./tmp/adjust_entr_select_w{weight}.jpg"), (entr_map2/entr_map2.max()*255).astype(int))
                print("==========================================")
        print("avg ratio: ", np.array(ratio_list).mean(), np.array(rratio_list).mean())
        return np.array(ratio_list).mean(), np.array(rratio_list).mean()

    def entr_map_from_image3(self, thresholds=[0,999], inverse=False, temperature=0.5):
        """ adjusted entr map, in-threshold: 1, out-threshold: 0 """
        print("Get entropy maps! 222 ")
        self.entr_map_dict = {}
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
                high_entr = high_entr.max() - high_entr
            # print("Adjusting porb map, high entr ratio: ", high_entr.sum() / (384*384))
            ratio_list.append(high_entr.sum() / (384*384))
            # assert entr_map2.sum() == 1, f"Got {entr_map2.shape, entr_map2.sum()}"
            self.entr_map_dict["{0:03d}".format(i+1)] = high_entr
            if i == 0:
                torchvision_save(torch.Tensor(high_entr.copy() / high_entr.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
            # if i == 0:
            #     cv2.imwrite(tfilename(self.runs_dir, f"./tmp/adjust_entr_select_{i}.jpg"), (high_entr/high_entr.max()*255).astype(int))
            #     print("==========================================")
        print("avg ratio: ", np.array(ratio_list).mean())
        return np.array(ratio_list).mean()

    def entr_map_from_image4(self, thresholds=[0,999], inverse=False, temperature=0.5):
        """ adjusted entr map, in-threshold: 0.5, out-threshold: 1 """
        print("Get entropy maps! 222 ")
        self.entr_map_dict = {}
        ratio_list = []
        for i in range(150):
            name = str(i+1) + ".npy"
            path = tfilename(self.entr_map_dir, name)
            if not os.path.exists(path):
                raise NotImplementedError
            else:
                entr_map = np.load(path)
            high_entr = np.ones_like(entr_map)
            high_entr[np.where(entr_map<thresholds[0])] = 0.5
            high_entr[np.where(entr_map>thresholds[1])] = 0.5
            assert entr_map.shape == (384, 384)
            if inverse:
                high_entr = 0.5 / high_entr
            ratio_list.append(high_entr.sum() / (384*384))
            self.entr_map_dict["{0:03d}".format(i+1)] = high_entr
            if i == 0:
                torchvision_save(torch.Tensor(high_entr.copy() / high_entr.max()),
                                 tfilename(self.runs_dir, name[:-4] + ".png"))
        print("avg ratio: ", np.array(ratio_list).mean())
        return np.array(ratio_list).mean()

    def select_point_from_entr_map(self, entr_map, size=(192, 192)):
        size_x, size_y = entr_map.shape
        assert size_x == size[0]
        assert size_y == size[1]
        chosen_x1 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y1 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        chosen_x2 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y2 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        if (entr_map[chosen_x1, chosen_y1]) * np.random.random() > (entr_map[chosen_x2, chosen_y2]) * np.random.random():
            return chosen_x1, chosen_y1
        else:
            return chosen_x2, chosen_y2

    def select_point_from_probmap_hard(self, prob_map, size=(384,384)):
        prob = rearrange(prob_map, "h w -> (h w)")
        prob = prob / prob.sum()
        loc = np.random.choice(a=self.xy, size=1, replace=True, p=prob)[0]
        return loc // self.size[1], loc % self.size[1]

    def retfunc2(self, index):
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
            raw_h, raw_w = self.select_point_from_probmap_hard(self.entr_map_dict[item['ID']], size=self.size)
        else:
            raw_h, raw_w = self.select_point_from_entr_map(self.entr_map_dict[item['ID']], size=self.size)

        # entr_value
        entr_value = self.entr_map_dict[item['ID']][raw_h, raw_w]

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
        assert b_left <= b_right
        assert b_top <= b_bot
        left = np.random.randint(b_left, b_right) if b_left < b_right else b_left
        top = np.random.randint(b_top, b_bot) if b_top < b_bot else b_top

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
        # chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size
        chosen_h = torch.div(temp.argmax(), patch_size, rounding_mode="trunc")
        chosen_w = temp.argmax() % patch_size
        return {'raw_imgs': item['image'], 'crop_imgs': crop_imgs, 'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'chosen_loc': torch.LongTensor([chosen_h, chosen_w]), 'ID': item['ID'],
                'entr': entr_value}

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

