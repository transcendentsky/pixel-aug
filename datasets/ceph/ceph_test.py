import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.utils.data as data
from PIL import Image
# from utils.augmentation import *
from utils.augmentation import augment_dict, apply_augment, get_augment
import cv2


class Test_Cephalometric(data.Dataset):
    def __init__(self, pathDataset, mode="subtest", size=384, R_ratio=0.05, wo_landmarks=False, ret_dict=False, preprocess=False, default_oneshot_id=114):
        # np.random.seed()
        self.num_landmark = 19
        self.size = size if isinstance(size, list) else [size, size]
        # min_size = min(size)
        self.Radius = int(max(self.size) * R_ratio)
        print("The sizes are set as ", self.size)
        self.original_size = [2400, 1935]
        self.preprocess = preprocess
        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')

        self.list = list()

        if mode == 'Train':
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
            end = 158
        elif mode == 'oneshot_debug':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = default_oneshot_id + 1
            end = default_oneshot_id + 1
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0], [1]),  # 0.5
        ])
        # self.transform2 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5]),
        # ])

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        self.mode = mode
        self.base = 16
        self.wo_landmarks = wo_landmarks
        self.ret_dict = ret_dict

        # self.transform_preprocess = transforms.Compose([        ])

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
            landmark[i] = int(landmark[i] * self.size[1-i] / self.original_size[1-i])
        return landmark

    def __getitem__(self, index, **kwargs):
        if self.preprocess == True:
            return self.return_func_1(index)
        else:
            return self.return_func_0(index)

    def return_func_1(self, index):
        # wtf
        item = self.list[index]

        pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
        pil_img = Image.open(pth_img).convert('RGB')
        preprocess_names = ['AutoContrast', 'Contrast', 'Equalize', 'Posterize2', ]
        preprocess_level = [0.7,0.7,0.7,1.0]
        pil_img = pil_img.resize((384,384))
        # import ipdb; ipdb.set_trace()
        # aug_names = list(augment_dict.keys())
        # name = aug_names[0]
        # pil_img.save("debug_ori.png")
        # pil_img2 = pil_img.transform(pil_img.size, Image.AFFINE, (1, 0.3, 0, 0, 1, 0))
        # pil_img2.save("tmp_visual/debug_a2.png")
        # import ipdb; ipdb.set_trace()

        return_dict = {}
        for name, level in zip(preprocess_names, preprocess_level):
            pil_img2 = apply_augment(pil_img, name=name, level=level) # , deform_map
            # pil_img2.save(tfilename("tmp_visual", f"debug_{name}.png"))
            return_dict[f'img_{name}'] = self.transform2(pil_img2)
        # import ipdb; ipdb.set_trace()
        return_dict['img'] = self.transform2(pil_img)

        if self.wo_landmarks:
            return return_dict

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))

        return {**return_dict, "landmark_list":landmark_list, "name": item['ID'] + '.bmp', "index": index}

    def return_func_0(self, index):
        item = self.list[index]

        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))

        if self.ret_dict:
            return {'image': item['image'], 'index':index}

        if self.wo_landmarks:
            return item['image']

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))

        return {"img":item['image'], "landmark_list":landmark_list, "name": item['ID'] + '.bmp'}


    def return_func_pure(self, index):
        item = self.list[index]
        pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
        im = cv2.imread(pth_img, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, self.size)

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))

        return {"img": im, "landmark_list": landmark_list, "name": item['ID'] + '.bmp'}


    def __len__(self):
        return len(self.list)


def test_prob_map(logger, config, *args, **kwargs):
    from utils.utils import visualize
    from torchvision.utils import save_image
    # from utils.augmentation import *
    id_oneshot = 26
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", preprocess=True)
    data = testset.__getitem__(id_oneshot, preprocess=True)


if __name__ == "__main__":
    from torchvision.utils import save_image
    from tutils import tfilename, trans_init, trans_args
    args = trans_args()
    logger, config = trans_init(args, file=__file__)
    test_prob_map(logger, config)
    # eval(args.func)(logger, config)
    # test_prob_map()