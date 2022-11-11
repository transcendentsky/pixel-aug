import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.utils.data as data
from PIL import Image
from .augment import cc_augment

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
    def __init__(self, pathDataset, mode, size=[384, 384], be_consistency=False, patch_size=192, pre_crop=False ,rand_psize=False):

        self.size = size
        self.original_size = [2400, 1935]

        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')
        self.patch_size = patch_size
        self.pre_crop = pre_crop
        if rand_psize:
            self.rand_psize = rand_psize
            self.patch_size = -1

        self.list = list()

        if mode == 'Oneshot':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        else:
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400

        self.pre_trans = transforms.Compose([transforms.RandomCrop((int(2400*0.8), int(1935*0.8)), padding=2)])

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

        num_repeat = 10
        if mode == 'Train':
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)

        self.mode = mode
        self.base = 16
        self.be_consistency = be_consistency

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 8) * 32

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark

    def __getitem__(self, index):
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
        margin_x = np.random.randint(0, self.size[0] - patch_size)
        margin_y = np.random.randint(0, self.size[0] - patch_size)
        crop_imgs = augment_patch(item['image']\
            [:, margin_y:margin_y+patch_size, margin_x:margin_x+patch_size],\
                self.aug_transform)

        chosen_x_raw = np.random.randint(int(0.1*patch_size), int(0.9*patch_size))
        chosen_y_raw = np.random.randint(int(0.1*patch_size), int(0.9*patch_size))
        raw_y, raw_x = chosen_y_raw+margin_y, chosen_x_raw+margin_x

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y_raw, chosen_x_raw] = 1
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        # print(chosen_y, chosen_x)
        chosen_y, chosen_x = temp.argmax() // patch_size, temp.argmax() % patch_size

        to_PIL(item['image']).save('Raw.png')
        to_PIL(crop_imgs).save('Crop.png')

        if self.be_consistency:
            crop_imgs_aug = augment_patch(item['image']\
                [:, margin_y:margin_y+patch_size, margin_x:margin_x+patch_size],\
                    self.aug_transform)
            temp = torch.zeros([1, patch_size, patch_size])
            temp[:, chosen_y_raw, chosen_x_raw] = 1
            temp = cc_augment(torch.cat([crop_imgs_aug, temp], 0))
            crop_imgs_aug = temp[:3]
            temp = temp[3]
            chosen_y_aug, chosen_x_aug = temp.argmax() // patch_size, temp.argmax() % patch_size
            return item['image'], crop_imgs, chosen_y, chosen_x, raw_y, raw_x, crop_imgs_aug, chosen_y_aug, chosen_x_aug
        # print(chosen_y, chosen_x)
        # import ipdb; ipdb.set_trace()
        return item['image'], crop_imgs, chosen_y, chosen_x, raw_y, raw_x

    def __len__(self):

        return len(self.list)


def test_Ce_img():
    from utils import visualize
    from tqdm import tqdm
    # hamming_set(9, 100)
    test = Cephalometric('../../dataset/Cephalometric', 'Train', pre_crop=True)
    for i in tqdm(range(2, 3)):
        item, crop_imgs, chosen_y, chosen_x, raw_y, raw_x = test.__getitem__(i)
        vis1 = visualize(item.unsqueeze(0), [[raw_x, raw_y]], [[raw_x, raw_y]])
        vis1.save(f"imgshow/train_pre_crop_{i}.jpg")
        vis2 = visualize(crop_imgs.unsqueeze(0), [[chosen_x, chosen_y]], [[chosen_x, chosen_y]])
        vis2.save(f"imgshow/train_pre_crop_{i}_crop.jpg")
        print("logging ", item.shape)
        print("crop", crop_imgs.shape)
        # for i in range(9):
        #     reimg = inv_trans(crop_imgs_1[:,:,:,i].transpose((1,2,0))).numpy().transpose((1,2,0))*255
        #     print(reimg.shape)
        #     cv2.imwrite(f"tmp/reimg_{i}.jpg", reimg.astype(np.uint8))
        import ipdb;
        ipdb.set_trace()
    print("pass")

class Test_Cephalometric(data.Dataset):
    def __init__(self, pathDataset, mode, size=[384, 384], id_oneshot=126, pre_crop=False):

        self.num_landmark = 19
        self.size = size
        if pre_crop:
            self.size[0] = 480 #int(size[0] / 0.8)
            self.size[1] = 480 #int(size[1] / 0.8)
        print("The sizes are set as ", self.size)
        self.original_size = [2400, 1935]

        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')

        self.list = list()

        if mode == 'Oneshot':
            print("One shot ID: ", id_oneshot)
            self.pth_Image = os. path.join(self.pth_Image, 'TrainingData')
            start = id_oneshot
            end = id_oneshot
        elif mode == 'Fewshots':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = int(150*0.25)
        elif mode == 'Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        else:
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400

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
            landmark[i] = int(landmark[i] * self.size[1-i] / self.original_size[1-i])
        return landmark

    def __getitem__(self, index):
        np.random.seed()
        item = self.list[index]

        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
            # print("??2,", item['image'].shape)

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark(landmark))

        if self.mode not in ['Oneshot', 'Fewshots']:
            # print("??, ", item['image'].shape)
            return item['image'], landmark_list

        template_patches = torch.zeros([self.num_landmark, 3, 192, 192])
        landmark_list2 = []
        for id, landmark in enumerate(landmark_list):
            left = min(max(landmark[0] - 96, 0), self.size[0]-192)
            bottom = min(max(landmark[1] - 96, 0), self.size[0]-192)
            template_patches[id] = item['image'][:, bottom:bottom+192, left:left+192]
            # landmark_list2[id] = [landmark[0] - left, landmark[1] - bottom]
            landmark_list2.append([landmark[0] - left, landmark[1] - bottom])
        return item['image'], landmark_list2, template_patches, landmark_list

    def __len__(self):
        return len(self.list)

def test_head_set():
    dataset = Test_Cephalometric('../../dataset/Cephalometric/', mode="Oneshot", pre_crop=True)
    item, landmark_list, template_patches = dataset.__getitem__(0)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # hamming_set(9, 100)
    # test = Cephalometric('../../dataset/Cephalometric', 'Oneshot')
    # for i in range(1):
    #     test.__getitem__(i)
    # print("pass")
    # test_Ce_img()
    test_head_set()