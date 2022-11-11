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
import math
from tutils import torchvision_save
from torch import nn
import copy


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
    def __init__(self, pathDataset, mode="Train", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None,
                 retfunc=2, use_prob=False, ret_name=False, be_consistency=False,
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
        elif mode == 'subtest':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 171
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

        self.init_mask()
        self.pool_fn = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    # def set_rand_psize(self):
    #     self.patch_size = np.random.randint(6, 8) * 32

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
        # print("====== Save Prob map to ./imgshow")
        # cv2.imwrite(f"imgshow/prob_map_ks{kernel_size}.jpg", (prob_map * 255).astype(np.uint8))
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

    def retfunc2(self, index):
        raw_h, raw_w = self.select_point_from_prob_map(self.prob_map, size=self.size)
        return self._retfunc_part2(index, raw_h, raw_w)

    def _retfunc_part2(self, index, raw_h, raw_w):
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
        # prob_map = self.prob_map_from_landmarks(self.size)
        # pad_scale = 0.05
        # raw_w = np.random.randint(int(pad_scale * self.size[0]), int((1 - pad_scale) * self.size[0]))
        # raw_h = np.random.randint(int(pad_scale * self.size[1]), int((1 - pad_scale) * self.size[1]))

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
        # chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size
        chosen_h = torch.div(temp.argmax(), patch_size, rounding_mode="trunc")
        chosen_w = temp.argmax() % patch_size

        gaussian_mask = self.gen_gaussian_mask([raw_h, raw_w])
        assert gaussian_mask[raw_h, raw_w] == 1.0, f"Got {gaussian_mask[raw_h, raw_w]}, max: {gaussian_mask.max()}"
        gmask = gaussian_mask.unsqueeze(0)
        gmask_dict = {} # 192 96 48 24 12
        for i in range(5):
            gmask = self.pool_fn(gmask)
            assert gmask.shape[-1] == 384 // 2**(i+1), f"Got {gmask.shape}, should be {384} // {2**(i+1)}"
            gmask_dict[f"gmask_{i}"] = 1 - (gmask[0] - gmask[0].min()) / (gmask[0].max() - gmask[0].min())
            # testloc_h, testloc_w = raw_h // 2**(i+1), raw_w // 2**(i+1)
            # if gmask_dict[f"gmask_{i}"][testloc_h, testloc_w] < 1.0:
            #     torchvision_save(gaussian_mask, "ogmask.png")
            #     torchvision_save(gmask_dict[f"gmask_{i}"], "debug.png")
            #     import ipdb; ipdb.set_trace()
            # assert gmask_dict[f"gmask_{i}"][testloc_h, testloc_w] >= 1.0, f'Got {gmask_dict[f"gmask_{i}"][testloc_h, testloc_w]}, and max:{gmask_dict[f"gmask_{i}"].max()}, level:{i}'

        return {'raw_imgs': item['image'],
                'crop_imgs': crop_imgs,
                'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'chosen_loc': torch.LongTensor([chosen_h, chosen_w]),
                'ID': item['ID'],
                **gmask_dict}

    def retfunc3(self, index):
        data = self.retfunc1(index)
        landmarks = data['raw_loc']
        mask, offset_y, offset_x = self.gen_mask(landmarks)
        data['mask'] = mask
        data['offset_y'] = offset_y
        data['offset_x'] = offset_x
        return data

    def init_mask(self):
        # gen mask
        self.Radius = 19 # int(max(self.size) * 0.05)
        mask = torch.zeros(2*self.Radius+1, 2*self.Radius+1, dtype=torch.float)
        gaussian_mask = torch.zeros(2*self.Radius+1, 2*self.Radius+1, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i - self.Radius, j - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1
                    gaussian_mask[i][j] = math.exp(-1 * math.pow(distance, 2) / \
                                       math.pow(8, 2))
        # gen offset
        self.mask = mask
        self.gaussian_mask = gaussian_mask
        self.gaussian_mask[19,19] += 0.2
        self.gaussian_mask /= 1.2
        assert gaussian_mask[19, 19] == 1.0, f"Got {gaussian_mask[19,19]}"
        torchvision_save(gaussian_mask, "gaussian_mask.png")

        self.offset_x = torch.zeros(2*self.Radius+1, 2*self.Radius+1, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius+1, 2*self.Radius+1, dtype=torch.float)
        for i in range(2*self.Radius+1):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

    def gen_gaussian_mask(self, loc):
        landmark = copy.deepcopy(loc)
        landmark[0] += self.Radius
        landmark[1] += self.Radius
        y, x = self.size[0] + 2*self.Radius, self.size[1] + 2*self.Radius
        mask = torch.zeros((y, x), dtype=torch.float)
        margin_y_bottom = max(0, landmark[0] - self.Radius)
        margin_y_top = min(y, landmark[0] + self.Radius + 1)
        margin_x_left = max(0, landmark[1] - self.Radius)
        margin_x_right = min(x, landmark[1] + self.Radius + 1)
        # corner_left_horizontal = True if landmark[0] - self.Radius < 0 else False
        # corner_up_vertical = True if landmark[1] - self.Radius < 0 else False
        # if corner_left_horizontal
        mask[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = self.gaussian_mask
        mask = mask[self.Radius:self.Radius+self.size[0], self.Radius:self.Radius+self.size[1]]
        assert mask.shape == (384, 384), f"Got {mask.shape}"
        torchvision_save(mask, "mask.png")
        assert mask[loc[0], loc[1]] == 1.0, f"Got {mask[loc[0], loc[1]]}"

        return mask

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
        if self.retfunc == 2:
            return self.retfunc2(index)
        elif self.retfunc == 3:
            return self.retfunc3(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.list)


class Test_Cephalometric(data.Dataset):
    def __init__(self, pathDataset, mode, size=[384, 384], return_mode="simple", id_oneshot=1, pre_crop=False):

        self.num_landmark = 19
        self.size = size
        if pre_crop:
            self.size[0] = 480  # int(size[0] / 0.8)
            self.size[1] = 480  # int(size[1] / 0.8)
        print("The sizes are set as ", self.size)
        self.original_size = [2400, 1935]
        self.Radius = int(max(self.size) * 0.05)

        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')

        self.list = list()
        self.return_mode = return_mode

        if mode == 'Oneshot':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = id_oneshot
            end = id_oneshot
        elif mode == 'subtest':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 20
        elif mode == 'Train':
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
        else:
            raise ValueError

        normalize = transforms.Normalize([0.5], [0.5])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        self.mode = mode
        self.base = 16

        print(f"Initializing Datasets: (split:'{mode}') (len:({len(self)})) ")

    def get_shots(self, n=1):
        if self.mode != 'Fewshots':
            raise ValueError(f"Got mode={self.mode}")

        item_list = []
        lm_list_list = []
        tp_list_list = []
        for i in range(n):
            item, landmark_list, template_patches = self.__getitem__(i)
            item_list.append(item)
            lm_list_list.append(landmark_list)
            tp_list_list.append(template_patches)
        return item_list, lm_list_list, tp_list_list

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[1 - i] / self.original_size[1 - i])
        return landmark

    def __getitem__(self, index):
        return self.retfunc_old(index)

    def retfunc_old(self, index):
        np.random.seed()
        item = self.list[index]

        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
            # print("??2,", item['image'].shape)

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID'] + '.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID'] + '.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5 * (int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))

        if self.return_mode in ['simple']:
            return {"raw_imgs":item['image'], "landmark_list": landmark_list}

        template_patches = torch.zeros([self.num_landmark, 3, 192, 192])
        for id, landmark in enumerate(landmark_list):
            left = min(max(landmark[0] - 96, 0), self.size[0] - 192)
            bottom = min(max(landmark[1] - 96, 0), self.size[0] - 192)
            template_patches[id] = item['image'][:, bottom:bottom + 192, left:left + 192]
            landmark_list[id] = [landmark[0] - left, landmark[1] - bottom]
        chosen_loc = [[96,96]] * self.num_landmark
        return {"raw_imgs": item['image'], "crop_imgs": template_patches, "landmark_list": landmark_list, "chosen_loc": chosen_loc}

    def ref_landmarks(self, index):
        np.random.seed()
        item = self.list[index]
        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID'] + '.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID'] + '.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5 * (int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))
        return landmark_list

    def __len__(self):
        return len(self.list)


def test_head_set(logger, config, *args, **kwargs):
    dataset = Test_Cephalometric('../../dataset/Cephalometric/', mode="Oneshot", pre_crop=True)
    item, landmark_list, template_patches = dataset.__getitem__(0)
    import ipdb;
    ipdb.set_trace()


def test_prob_map(logger, config, *args, **kwargs):
    from utils.utils import visualize
    from torchvision.utils import save_image
    id_oneshot = 26
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    # item, landmark_list, template_patches = testset.__getitem__(0)
    landmark_list = testset.ref_landmarks(id_oneshot)
    data = testset.__getitem__(id_oneshot)

    trainset = Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False, ref_landmark=landmark_list, retfunc=2, use_prob=True)
    # trainset._retfunc_part2(0, 1, 1)
    # import ipdb; ipdb.set_trace()
    data = trainset.__getitem__(id_oneshot)
    print("landmark list", landmark_list)
    prob_map = trainset.prob_map_from_landmarks()
    print("prob map max min", np.max(prob_map), np.min(prob_map))
    save_image(prob_map, f"tmp/prob_map_{id_oneshot}.png")
    import ipdb; ipdb.set_trace()

    cv2.imwrite(tfilename(config['base']['runs_dir'], f"/prob_map_{id_oneshot}.png"), (prob_map * 255).astype(np.uint8))
    # save_image(data['raw_imgs'], tfilename(config['base']['runs_dir'], f"id_{id_oneshot}.png"))
    image_pil = visualize(data['raw_imgs'], data['landmark_list'], data['landmark_list'], num=19)
    image_pil.save(tfilename(config['base']['runs_dir'], f"id_{id_oneshot}.png"))

    # ----------------------------------------------------------------------
    item, crop_imgs, chosen_h, chosen_w, raw_h, raw_w = trainset.retfunc2(0)
    i = 0
    vis1 = visualize(item.unsqueeze(0), [[raw_w, raw_h]], [[raw_w, raw_h]])
    vis1.save(f"imgshow/train_prob_{i}.jpg")
    vis2 = visualize(crop_imgs.unsqueeze(0), [[chosen_w, chosen_h]], [[chosen_w, chosen_h]])
    vis2.save(f"imgshow/train_prob_{i}_crop.jpg")
    print("logging ", item.shape)
    print("crop", crop_imgs.shape)
    # import ipdb; ipdb.set_trace()




if __name__ == "__main__":
    from torchvision.utils import save_image
    from tutils import tfilename, trans_init, trans_args
    args = trans_args()
    logger, config = trans_init(args, file=__file__)
    eval(args.func)(logger, config)
    # test_prob_map()