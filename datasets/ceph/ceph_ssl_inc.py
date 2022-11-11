"""
    Incremental detection
    Fully supervised +　SSL

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
    def __init__(self, pathDataset, mode="Train", size=384, patch_size=192,
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
        size_y, size_x = prob_map.shape
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

    def retfunc3(self, index):
        """
         Two images with landmarks via the SSL style
        """
        raise NotImplementedError

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
        raw_w = np.random.randint(int(pad_scale * self.size[0]), int((1 - pad_scale) * self.size[0]))
        raw_h = np.random.randint(int(pad_scale * self.size[1]), int((1 - pad_scale) * self.size[1]))
        res_dict = self._process(item['image'], (raw_h, raw_w), index)
        return res_dict


    def retfunc2(self, index):
        """
        New Point Choosing Function with 'PROB MAP'
        """
        np.random.seed()
        item = self.list[index]
        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
        raw_w, raw_h = self.select_point_from_prob_map(self.prob_map, size=self.size)
        res_dict = self._process(item['image'], (raw_h, raw_w), index)
        return res_dict

    def _process(self, img, raw_loc, index, img2=None):
        raw_h, raw_w = raw_loc
        patch_size = self.patch_size
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
        chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size
        return {'raw_imgs': img,
                'crop_imgs': crop_imgs,
                'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'chosen_loc': torch.LongTensor([chosen_h, chosen_w]),
                'ID': index}

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

    # def gen_mask(self, loc, img_shape=[384, 384]):
    #     # GT, mask, offset
    #     loc = np.array(loc)
    #     assert loc.ndim == 2, f"Got loc {loc}"
    #     num_landmark = loc.shape[0]
    #     y, x = self.size[0], self.size[1]
    #     mask = torch.zeros((num_landmark, y, x), dtype=torch.float)
    #     offset_x = torch.zeros((num_landmark, y, x), dtype=torch.float)
    #     offset_y = torch.zeros((num_landmark, y, x), dtype=torch.float)
    #
    #     for i in range(num_landmark):
    #         landmark = loc[i]
    #         margin_x_left = max(0, landmark[0] - self.Radius)
    #         margin_x_right = min(x, landmark[0] + self.Radius)
    #         margin_y_bottom = max(0, landmark[1] - self.Radius)
    #         margin_y_top = min(y, landmark[1] + self.Radius)
    #
    #         mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
    #             self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
    #         offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
    #             self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
    #         offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
    #             self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
    #     return mask, offset_y, offset_x
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
            return self.retfunc1(index)
        elif self.retfunc == 2:
            return self.retfunc2(index)
        elif self.retfunc == 3:
            return self.retfunc3(index)
        elif self.retfunc == 0:
            return self.retfunc_old(index)
        else:
            raise ValueError

    def retfunc_old(self, index):
        # raise ValueError
        np.random.seed()
        item = self.list[index]

        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            if self.pre_crop:
                item['image'] = self.transform(self.pre_trans(Image.open(pth_img).convert('RGB')))
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))

        # Crop 192 x 192 Patch
        # patch_size = int(0.5 * self.size[0])
        patch_size = self.patch_size
        margin_w = np.random.randint(0, self.size[0] - patch_size)
        margin_h = np.random.randint(0, self.size[0] - patch_size)
        crop_imgs = augment_patch(item['image'] \
                                      [:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size], \
                                  self.aug_transform)

        chosen_x_raw = np.random.randint(int(0.1 * patch_size), int(0.9 * patch_size))
        chosen_y_raw = np.random.randint(int(0.1 * patch_size), int(0.9 * patch_size))
        raw_h, raw_w = chosen_y_raw + margin_h, chosen_x_raw + margin_w

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y_raw, chosen_x_raw] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        # print(chosen_h, chosen_w)
        chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size

        to_PIL(item['image']).save('Raw.png')
        to_PIL(crop_imgs).save('Crop.png')

        if self.be_consistency:
            crop_imgs_aug = augment_patch(item['image'] \
                                              [:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size], \
                                          self.aug_transform)
            temp = torch.zeros([1, patch_size, patch_size])
            temp[:, chosen_y_raw, chosen_x_raw] = 1
            temp = cc_augment(torch.cat([crop_imgs_aug, temp], 0))
            crop_imgs_aug = temp[:3]
            temp = temp[3]
            chosen_y_aug, chosen_x_aug = temp.argmax() // patch_size, temp.argmax() % patch_size
            return item['image'], crop_imgs, chosen_h, chosen_w, raw_h, raw_w, crop_imgs_aug, chosen_y_aug, chosen_x_aug
        # print(chosen_h, chosen_w)
        # import ipdb; ipdb.set_trace()
        return item['image'], crop_imgs, chosen_h, chosen_w, raw_h, raw_w

    def __len__(self):
        return len(self.list)


class Test_Cephalometric(data.Dataset):
    def __init__(self, pathDataset, mode, size=[384, 384], id_oneshot=1, pre_crop=False):

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

        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        self.mode = mode
        self.base = 16

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

        if self.mode not in ['Oneshot']:
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

    trainset = Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False, ref_landmark=landmark_list, retfunc=2)
    # data = trainset.__getitem__(id_oneshot)
    print("landmark list", landmark_list)
    prob_map = trainset.prob_map_from_landmarks()
    print("prob map max min", np.max(prob_map), np.min(prob_map))

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