"""
    See jupyter notebook
"""
import numpy as np
from tutils import tfilename
import os
import cv2
from sklearn.metrics import mutual_info_score
# from scipy.misc import imread
import numpy as np
import cv2
from skimage.exposure import histogram
from scipy.stats import entropy

import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns


def draw_bar(x, y):
    fig = sns.lineplot(x=x, y=y, label="tt")
    plt.savefig("test_debug.png")
    plt.close()
    # print("Drawed img: ", fname)


def ana_aug():
    # lm0: min 0.66, max 1.80
    d = {
        "baseline": {
            "cj_brightness": 0.15,  # 0.15
            "cj_contrast": 0.25,  # 0.25
            "cj_saturation": 0.,  # 0.
            "cj_hue": 0.,  # 0
            "mre": 2.45,
            "mi": -0.1,
        },
        "v2_br_0.5": {
            "cj_brightness": 0.5, # 0.15
            "cj_contrast": 0.25, # 0.25
            "cj_saturation": 0., # 0.
            "cj_hue": 0., # 0
            "mre": 2.43,
            "mi": -0.1,
        },
        "v2_br_0.9": {
            "cj_brightness": 1.6,  # 0.15
            "cj_contrast": 1.2,  # 0.25
            "cj_saturation": 0.,  # 0.
            "cj_hue": 0.,  # 0
            "mre": 2.39,
            "mi": -0.1,
        },
        "v2_br_ct": {

        },
        "v2_br_ct_large": {},
        "v2_ct_0.9": {},

    }


def test_mi_of_aug_img():
    """ from test_all_mi.py """
    import torch
    import torchvision
    from torchvision import transforms
    from PIL import Image
    import torchvision.transforms.functional as F

    im = cv2.imread("001.bmp", cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (384, 384))
    lm = [322, 311]
    ps_half = 32
    patch = im[lm[0]-ps_half:lm[0]+ps_half, lm[1]-ps_half:lm[1]+ps_half]

    def get_fea(patch):
        fea = np.zeros((256,))
        hist, idx = histogram(patch, nbins=256)
        for hi, idi in zip(hist, idx):
            # print(hi, idi, i, j)
            fea[idi] = hi
        return fea

    fea1 = get_fea(patch)
    cv2.imwrite("patch1.jpg", patch)

    # fn_aug = transforms.ColorJitter(brightness=0.9)
    patch_aug = Image.fromarray(patch)
    patch_aug = F.adjust_brightness(patch_aug, 9)
    patch_aug = np.array(patch_aug)
    cv2.imwrite("patch2.jpg", patch_aug)
    fea2 = get_fea(patch_aug)

    mi0 = mutual_info_score(fea1, fea1)
    mi = mutual_info_score(fea1, fea2)
    print(mi, mi0)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    # ana_aug()
    test_mi_of_aug_img()