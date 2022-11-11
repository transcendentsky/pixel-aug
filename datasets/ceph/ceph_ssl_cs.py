"""
    Keep Consistency of proxy task and target task

    Used with ssl_cs.py
"""


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
    def __init__(self, pathDataset,
                 mode="Train",
                 size=384,
                 R_ratio=0.05,
                 patch_size=192,
                 pre_crop=False,
                 rand_psize=False,
                 ref_landmark=None,
                 prob_map=None,
                 retfunc=1,
                 use_prob=False,
                 ret_name=False,
                 be_consistency=False,
                 num_repeat=10):
        assert not (ref_landmark is not None and prob_map is not None), \
            f"Got both ref_landmark: \n{ref_landmark} \nand prob_map: \n{prob_map}"

        self.num_landmark = 19
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
            if prob_map is None:
                self.prob_map = self.prob_map_from_landmarks(self.size)


        # gen mask
        self.Radius = int(max(size) * R_ratio)
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

        self.offset_w = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_h = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_w[:, i] = self.Radius - i
            self.offset_h[i, :] = self.Radius - i
        self.offset_w = self.offset_w * self.mask / self.Radius
        self.offset_h = self.offset_h * self.mask / self.Radius


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
        # raw_w, raw_h = self.select_point_from_prob_map(self.prob_map, size=self.size)
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
        cimg = item['image'][:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size]
        crp_imgs = augment_patch(cimg, self.aug_transform)
        chosen_w, chosen_h = raw_w - margin_w, raw_h - margin_h

        raw_mask, raw_offset_h, raw_offset_w = self.get_mask_from_lm(item['image'], (raw_h, raw_w))
        crp_mask, crp_offset_h, crp_offset_w = self.get_mask_from_lm(crp_imgs, (chosen_h, chosen_w))

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_h, chosen_w] = 1
        temp = cc_augment(torch.cat([crp_imgs, temp, crp_mask, crp_offset_h, crp_offset_w], 0))
        crp_imgs = temp[:3]
        locs = temp[3]
        crp_mask = temp[4]
        crp_offset_h = temp[5]
        crp_offset_w = temp[6]

        chosen_h, chosen_w = locs.argmax() // patch_size, locs.argmax() % patch_size

        return {'raw_imgs': item['image'], 
                'crp_imgs': crp_imgs,
                'raw_loc': torch.LongTensor([raw_h, raw_w]), 
                'chosen_loc': torch.LongTensor([chosen_h, chosen_w]),
                'raw_mask': raw_mask, 'raw_offset_h': raw_offset_h, 'raw_offset_w': raw_offset_w,
                'crp_mask': crp_mask, 'crp_offset_h': crp_offset_h, 'crp_offset_w': crp_offset_w,
                'ID': item['ID']}
    
    def get_mask_from_lm(self, img, landmark):
        h, w = img.shape[-2], img.shape[-1]
        raw_h, raw_w = landmark
        mask = torch.zeros((1, h, w), dtype=torch.float)
        offset_w = torch.zeros((1, h, w), dtype=torch.float)
        offset_h = torch.zeros((1, h, w), dtype=torch.float)
        margin_w_left = max(0, raw_w - self.Radius)
        margin_w_right = min(w, raw_h + self.Radius)
        margin_h_bottom = max(0, raw_w - self.Radius)
        margin_h_top = min(h, raw_h + self.Radius)

        mask[:, margin_h_bottom:margin_h_top, margin_w_left:margin_w_right] = \
            self.mask[0:margin_h_top-margin_h_bottom, 0:margin_w_right-margin_w_left]
        offset_w[:, margin_h_bottom:margin_h_top, margin_w_left:margin_w_right] = \
            self.offset_w[0:margin_h_top-margin_h_bottom, 0:margin_w_right-margin_w_left]
        offset_h[:, margin_h_bottom:margin_h_top, margin_w_left:margin_w_right] = \
            self.offset_h[0:margin_h_top-margin_h_bottom, 0:margin_w_right-margin_w_left]
        assert len(mask.shape) == 3
        return mask, offset_h, offset_w

    def __getitem__(self, index):
        if self.retfunc == 1:
            return self.retfunc1(index)
        else:
            raise ValueError


if __name__ == '__main__':
    from tutils import trans_args, trans_init, tfilename
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default="configs/confing")
    args = trans_args()
    logger, config = trans_init(args)
    dataset = Cephalometric(config['dataset']['pth'])
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    for data in loader:
        key = 'raw_offset_h'
        save_image(data[key], f"tmp/{key}.png")
        key = 'raw_offset_w'
        save_image(data[key], f"tmp/{key}.png")
        key = 'crp_offset_h'
        save_image(data[key], f"tmp/{key}.png")
        key = 'crp_offset_w'
        save_image(data[key], f"tmp/{key}.png")
        key = 'raw_mask'
        save_image(data[key], f"tmp/{key}.png")
        key = 'crp_mask'
        save_image(data[key], f"tmp/{key}.png")
        import ipdb; ipdb.set_trace()


