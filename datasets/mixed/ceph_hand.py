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
    # image.save('raw.jpg')
    aug_image = aug_transform(image)
    # aug_image_PIL = to_PIL(aug_image)
    # aug_image_PIL.save('aug.jpg')
    # import ipdb; ipdb.set_trace()
    return aug_image



class CephAndHand_SSL(data.Dataset):
    """
    Self-supervised Learning method: Images only, no label
    """
    def __init__(self,
                 mode="Train",
                 size=384,
                 patch_size=192,
                 img_dir1=None,
                 img_dir2=None,
                 ):
        self.mode = mode
        self.size = size if not isinstance(size, int) else [size, size]
        self.patch_size = patch_size

        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self._init_paths()
        self._init_transforms()

    def _init_paths(self):
        mode = self.mode

        # Cephalometric
        self.img_dir1 = tfilename(self.img_dir1, "RawImage")
        if mode in ['Train']:
            self.img_dir1 = os.path.join(self.img_dir1, 'TrainingData')
            start = 1
            end = 150
        elif mode == 'Test1':
            self.img_dir1 = os.path.join(self.img_dir1, 'Test1Data')
            start = 151
            end = 300
        elif mode == 'Test2':
            self.img_dir1 = os.path.join(self.img_dir1, 'Test2Data')
            start = 301
            end = 400
        elif mode == 'Test1+2':
            self.img_dir1 = os.path.join(self.img_dir1, 'Test1Data')
            start = 151
            end = 400
        elif mode == 'subtest':
            self.img_dir1 = os.path.join(self.img_dir1, 'Test1Data')
            start = 151
            end = 171
        else:
            raise ValueError

        self.ceph_paths = []
        for i in range(start, end + 1):
            self.ceph_paths.append(['ceph', "{0:03d}".format(i) + '.bmp'])
            # self.ceph_paths.append(os.path.join(self.img_dir1, "{0:03d}".format(i) + '.bmp'))

        # Hand Xray
        self.img_dir2 = tfilename(self.img_dir2, "jpg")
        self.hand_paths = [x.name for x in os.scandir(self.img_dir2) if x.name.endswith(".jpg")]
        self.hand_paths.sort()
        self.train_list = self.hand_paths[300:]
        self.test_list = self.hand_paths[:300]
        self.hand_paths = [["hand", path] for path in self.train_list]
        # print(self.hand_paths)
        # label_path = tfilename(self.img_dir2, "all.csv")
        # self.landmarks = get_csv_content(label_path)
        # self.landmarks_test = self.landmarks[:300]
        # self.landmarks_train = self.landmarks[300:]

        self.data_paths = self.ceph_paths + self.hand_paths

    def _init_transforms(self):
        self.transform = transforms.Compose([transforms.Resize(self.size),
                                             transforms.RandomAffine(degrees=0, translate=(0, 0.1)),
                                             transforms.ColorJitter(brightness=0.15, contrast=0.25),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0], [1])])
        self.extra_aug_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.25)], p=0.5),
            transforms.ToTensor(),
            # AddGaussianNoise(0., 1.),
            transforms.Normalize([0], [1]),
        ])

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])

    def _load_original_data(self, index):
        np.random.seed()
        _d = self.data_paths[index]
        if _d[0] == "ceph":
            img_dir = self.img_dir1
        elif _d[0] == "hand":
            img_dir = self.img_dir2
        path = tfilename(img_dir, _d[1])
        img = Image.open(path).convert('RGB')
        return img

    def __getitem__(self, index):
        original_data = self._load_original_data(index)
        return self._process1(original_data, index)

    def _process1(self, img, index):
        """
                New Point Choosing Function without prob map
                """
        img = self.transform(img)

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
        cimg = img[:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size]
        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_w, chosen_h = raw_w - margin_w, raw_h - margin_h

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_h, chosen_w] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size

        return {'raw_imgs': img,
                'crop_imgs': crop_imgs,
                'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'chosen_loc': torch.LongTensor([chosen_h, chosen_w]),
                'ID': index}

    def __len__(self):
        return len(self.data_paths)


def usage():
    from torch.utils.data import DataLoader
    from utils.utils import visualize_std
    dataset = CephAndHand_SSL(img_dir1='/home1/quanquan/datasets/Cephalometric/', img_dir2='/home1/quanquan/datasets/hand/hand/')
    loader = DataLoader(dataset, batch_size=2)

    data0 = dataset.__getitem__(0)
    import ipdb; ipdb.set_trace()

    for data1 in loader:
        # save_image()
        import ipdb; ipdb.set_trace()
        break


if __name__ == "__main__":
    from torchvision.utils import save_image
    from tutils import tfilename, trans_init, trans_args
    # args = trans_args()
    # logger, config = trans_init(args, file=__file__)
    # eval(args.func)(logger, config)
    # test_prob_map()
    usage()