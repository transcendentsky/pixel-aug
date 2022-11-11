import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torchvision
from torchvision import transforms
from PIL import Image
from tutils import torchvision_save


aug_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.3, contrast=0.25,
                           saturation=0, hue=0),
   transforms.ToTensor(),
   transforms.Normalize([0], [1])
])
resizefn = transforms.Compose([transforms.Resize((384,384)),
                               transforms.ToTensor(),
                               transforms.Normalize([0], [1])])

pil_im = Image.open("001.bmp").convert('RGB')

ori = resizefn(pil_im)
torchvision_save(ori, "ori001.jpg")

pil2 = aug_transform(ori)
torchvision_save(pil2, "aba.jpg")

# im = cv2.imread("001.bmp", cv2.IMREAD_GRAYSCALE)
# # im = cv2.resize(im, (384, 384))
# print(im.shape)
# entr_img = entropy(im, disk(2000))
# entr_img = cv2.resize(entr_img, (384, 384))