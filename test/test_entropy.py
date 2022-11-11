import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.exposure import histogram
from sklearn.metrics import mutual_info_score
# from scipy.misc import imread
import numpy as np
import cv2
# from scipy.stats import entropy
from tutils import torchvision_save

# ----------------------
# im = cv2.imread("001.bmp", cv2.IMREAD_GRAYSCALE)
# im = im[:8, :8]
# print(im)
# print(im.shape)
# hist = histogram(im, nbins=2)
# print(hist)
# exit(0)

# dirpth = "/home1/quanquan/datasets/Cephalometric/RawImage/TrainingData/"
dirpth = "./"
# im = cv2.imread(dirpth + "001.bmp", cv2.IMREAD_GRAYSCALE)
im = cv2.imread("026.bmp", cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, (384, 384))
torchvision_save(torch.Tensor(im), "ori_026.png")
# exit()
print(im.shape)
cv2.imwrite("384_3167.jpg", cv2.resize(im, (384, 384)))
entr_img = entropy(im, disk(10))
entr_img = cv2.resize(entr_img, (384, 384))
torchvision_save(torch.Tensor(entr_img), "my_entr.png")

high_entr = np.ones_like(entr_img)
# high_entr[np.where(entr_img<4.5)] = 0
# high_entr[np.where(entr_img>5.75)] = 0
high_entr[np.where(entr_img<4)] = 0
high_entr[np.where(entr_img>4.57)] = 0
print(high_entr.shape)
# cv2.imwrite("entr_026.png", entr_img)
torchvision_save(torch.Tensor(entr_img), "entr_hand.png")
torchvision_save(torch.Tensor(high_entr), "high.png")
# def get_entropy():


# rng = np.random.default_rng()
#
# noise_mask = np.full((128, 128), 28, dtype=np.uint8)
# noise_mask[32:-32, 32:-32] = 30
#
# noise = (noise_mask * rng.random(noise_mask.shape) - 0.5
#          * noise_mask).astype(np.uint8)
# img = noise + 128
#
# entr_img = entropy(img, disk(10))
# print(img.shape)
#
# plt.imshow(entr_img)
# plt.show()
#
# fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
#
# img0 = ax0.imshow(noise_mask, cmap='gray')
# ax0.set_title("Object")
# ax1.imshow(img, cmap='gray')
# ax1.set_title("Noisy image")
# ax2.imshow(entr_img, cmap='viridis')
# ax2.set_title("Local entropy")
#
# fig.tight_layout()
