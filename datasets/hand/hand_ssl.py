import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import h5py, os, random, math, torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
from datasets.augment import cc_augment
import csv
import torchvision.transforms.functional as TF
from tutils import tfilename


def to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8).transpose(1, 2, 0))
    return images


def augment_patch(tensor, aug_transform):
    image = to_PIL(tensor)
    aug_image = aug_transform(image)
    return aug_image


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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


class HandXray(data.Dataset):
    # '/home/quanquan/hand/hand/histo-norm/'
    def __init__(self, pathDataset='/home/quanquan/hand/hand/',\
                 mode="Train", size=[384, 384],
                 extra_aug_loss=False, patch_size=192, rand_psize=False,
                 cj_brightness=0.25, cj_contrast=0.15,
                 retfunc=1, num_repeat=1):
        """
        extra_aug_loss: return an additional image for image augmentation consistence loss
        """
        self.size = size
        self.extra_aug_loss = extra_aug_loss  # consistency loss
        self.pth_Image = tfilename(pathDataset, "jpg")
        self.patch_size = patch_size
        self.num_repeat = num_repeat

        self.list = [x.path for x in os.scandir(self.pth_Image) if x.name.endswith(".jpg")]
        self.list.sort()
        # print(self.list)
        label_path = tfilename(pathDataset, "all.csv")
        self.landmarks = get_csv_content(label_path)
        self.test_list = self.list[:300]
        self.landmarks_test = self.landmarks[:300]
        self.train_list = self.list[300:]
        self.landmarks_train = self.landmarks[300:]

        if mode in ["Oneshot", "Train"]:
            self.istrain = True
        elif mode in ["Test1", "Test"]:
            self.istrain = False
        else:
            raise NotImplementedError

        normalize = transforms.Normalize([0.], [1.])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        self.transform_resize = transforms.Resize(self.size)
        self.transform_tensor = transforms.ToTensor()

        # self.extra_aug_transform = transforms.Compose([
        #     transforms.Resize(self.size),
        #     transforms.RandomApply([
        #         transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        #         transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast)], p=0.5),
        #     transforms.ToTensor(),
        #     # AddGaussianNoise(0., 1.),
        #     transforms.Normalize([0], [1]),
        # ])

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])

        self.retfunc = retfunc
        self.mode = mode
        self.base = 16

        if mode in ['Train', 'Oneshot']:
            self.loading_list = self.train_list
            self.real_len = len(self.loading_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if self.retfunc == 1:
            return self.retfunc1(index)
        else:
            raise ValueError

    def retfunc1(self, index):
        index = index % self.real_len
        np.random.seed()
        img, landmarks, spacing = self.get_data(index)
        return self._process1(index, img)

    def get_data(self, index):
        img_pil = Image.open(self.train_list[index]).convert('RGB')
        landmarks = None
        spacing = None
        return img_pil, landmarks, spacing

    def _process1(self, index, img):
        """
        New Point Choosing Function without prob map
        """
        img = self.transform(img)
        pad_scale = 0.05
        padding = int(pad_scale * self.size[0])
        patch_size = self.patch_size
        raw_h = np.random.randint(int(pad_scale * self.size[0]), int((1 - pad_scale) * self.size[0]))
        raw_w = np.random.randint(int(pad_scale * self.size[1]), int((1 - pad_scale) * self.size[1]))

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
        # chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size
        chosen_h = torch.div(temp.argmax(), patch_size, rounding_mode="trunc")
        chosen_w = temp.argmax() % patch_size

        return {'raw_imgs': img, 'crop_imgs': crop_imgs, 'raw_loc': torch.LongTensor([raw_h, raw_w]), 'chosen_loc': torch.LongTensor([chosen_h, chosen_w]), 'ID': index }

    def __len__(self):
        return len(self.loading_list) * self.num_repeat


class HandXrayLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(HandXrayLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        self.dataset.set_rand_psize()
        return super(HandXrayLoader, self).__iter__()


class TestHandXray(data.Dataset):
    def __init__(self, pathDataset='/home/quanquan/hand/hand/jpg/', label_path='/home/quanquan/hand/hand/all.csv',
                 mode=None, istrain=False, size=[384, 384], load_mod="img"):
        self.istrain = istrain
        self.size = size
        self.original_size = [1, 1]
        self.label_path = label_path
        self.num_landmark = 37  # for hand example

        self.pth_Image = os.path.join(pathDataset)

        self.list = np.array([x.path for x in os.scandir(self.pth_Image) if x.name.endswith(".jpg")])
        self.list.sort()
        # print(self.list)
        self.test_list = self.list[:20]
        self.train_list = self.list[20:]
        self.landmarks = get_csv_content(label_path)
        # for landmark in self.landmarks:
        #     print("landmark indx", landmark['index'])
        self.landmarks_test = self.landmarks[:20]
        self.landmarks_train = self.landmarks[20:]
        print("train_list: ", len(self.train_list))
        print("test_list : ", len(self.test_list))

        # Read all imgs to mem in advanced
        # self.train_items = [Image.open(x) for x in self.train_list]
        # self.test_items = [Image.open(x) for x in self.test_list]

        # transform for both test/train
        self.transform = transforms.Compose([
            transforms.Resize(self.size),

        ])

        self.simple_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0], [1]),
        ])

        # transform for only train
        transform_list = [
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.10, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.aug_transform = transforms.Compose(transform_list)

        # self.maxHammingSet = generate_hamming_set(9, 100)
        self.mode = mode

    def resize_landmark(self, landmark, img_shape):
        """
        landmark = ['index': index, 'landmark': [(x,y), (x,y), (x,y), ...]]
        """
        for i in range(len(landmark)):
            # print("[Trans Debug] ", landmark[i], img_shape)
            landmark[i][0] = int(landmark[i][0] * self.size[0] / float(img_shape[0]))
            landmark[i][1] = int(landmark[i][1] * self.size[1] / float(img_shape[1]))
        return landmark
        # for i in range(len(landmark)):
        #     landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        # return landmark

    def _get_original(self, index):
        item_path = self.test_list[index]
        ori_img = Image.open(item_path)
        img_shape = np.array(ori_img).shape
        item = self.simple_trans(ori_img.convert('RGB'))
        return item, self.landmarks_test[index]['landmark'],

    def get_oneshot_name(self, index):
        img_path = self.train_list[index]
        parent, name = os.path.split(img_path)
        return parent, name

    def get_one_shot(self, index=1, debug=False):  # some bugs in index=0
        img_path = self.train_list[index]
        print("oneshot img pth: ", img_path)
        ori_img = Image.open(img_path)
        # if self.transform is not None:
        img = self.transform(ori_img.convert('RGB'))
        shape = np.array(ori_img).shape[::-1]
        landmark_list = self.resize_landmark(self.landmarks_train[index]['landmark'], shape)
        if debug:  # The part above is correct!
            return img, landmark_list, shape  # disacrd the index

        ##############################################
        template_patches = torch.zeros([self.num_landmark, 3, 192, 192])
        for id, landmark in enumerate(landmark_list):
            left = min(max(landmark[0] - 96, 0), 192)
            bottom = min(max(landmark[1] - 96, 0), 192)
            template_patches[id] = img[:, bottom:bottom + 192, left:left + 192]
            landmark_list[id] = [landmark[0] - left, landmark[1] - bottom]
            # if id == 9:
            #     print(landmark)
            #     print(left, bottom)
            #     to_PIL(template_patches[id]).save('template.jpg')
        return img, landmark_list, template_patches

    def __getitem__(self, index):
        """
        landmark = {'index': index, 'landmark': [(x,y), (x,y), (x,y), ...]}
        """
        if self.istrain:
            item_path = self.train_list[index]
        else:
            item_path = self.test_list[index]
        ori_img = Image.open(item_path)
        img_shape = np.array(ori_img).shape[::-1]  # shape: (y, x) or (long, width)
        # if self.transform is not None:
        item = self.transform(ori_img.convert('RGB'))
        # import ipdb; ipdb.set_trace()

        if self.istrain:
            landmarks = self.landmarks_train[index]['landmark']
        else:
            landmarks = self.landmarks_test[index]['landmark']
        landmark = self.resize_landmark(landmarks, img_shape)  # [1:] for discarding the index
        assert landmark is not None, f"Got Landmarks None, {item_path}"
        assert img_shape is not None, f"Got Landmarks None, {item_path}"
        return item, landmark, img_shape

    def __len__(self):
        if self.istrain:
            return len(self.train_list)
        else:
            return len(self.test_list)


def histo_normalize(img, ref, name=None):
    out = np.zeros_like(img)
    _, _, colorChannel = img.shape
    for i in range(colorChannel):
        # print(i)
        hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
        hist_ref, _ = np.histogram(ref[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
        cdf_ref = np.cumsum(hist_ref)

        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
            out[:, :, i][img[:, :, i] == j] = idx
    if name is not None:
        cv2.imwrite(f'/home/quanquan/hand/hand/histo-norm/{name}', out)
        print(f'Save: {name}')


def app_histo_norm():
    # from tqdm import tqdm
    data_root = "/home/quanquan/hand/hand/jpg/"
    ref_path = data_root + "3143.jpg"
    ref = cv2.imread(ref_path)
    for x in os.scandir(data_root):
        if x.name.endswith(".jpg"):
            img = cv2.imread(x.path)
            histo_normalize(img, ref, name=x.name)


def test_HandXray_img():
    from utils import visualize
    from tqdm import tqdm
    # hamming_set(9, 100)
    test = HandXray(patch_size=208)
    for i in tqdm(range(2, 3)):
        item, crop_imgs, chosen_y, chosen_x, raw_y, raw_x = test.__getitem__(i)
        vis1 = visualize(item.unsqueeze(0), [[raw_x, raw_y]], [[raw_x, raw_y]])
        vis1.save(f"imgshow/train_{i}.jpg")
        vis2 = visualize(crop_imgs.unsqueeze(0), [[chosen_x, chosen_y]], [[chosen_x, chosen_y]])
        vis2.save(f"imgshow/train_{i}_crop.jpg")
        print("logging ", item.shape)
        print("crop", crop_imgs.shape)
        # for i in range(9):
        #     reimg = inv_trans(crop_imgs_1[:,:,:,i].transpose((1,2,0))).numpy().transpose((1,2,0))*255
        #     print(reimg.shape)
        #     cv2.imwrite(f"tmp/reimg_{i}.jpg", reimg.astype(np.uint8))
        import ipdb;
        ipdb.set_trace()
    print("pass")


def test3():
    from utils import visualize
    test = TestHandXray()
    img, landmark_list, shape = test.get_one_shot(1, True)
    print(landmark_list)
    vis = visualize(img.unsqueeze(0), landmark_list, landmark_list)
    vis.save(f'imgshow/refer2_{1}.jpg')
    print("dasdas")


def test4():
    from utils import visualize
    test = TestHandXray()
    dataloader_1 = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)
    for img, landmark, shape in dataloader_1:
        vis = visualize(img, landmark, landmark)
        vis.save(f'imgshow/dataloader_{1}.jpg')
        print("save")


# if __name__ == "__main__":
#     # hamming_set(9, 100)
#     test_HandXray_img()
    # test2()
    # test4()
    # app_histo_norm()
