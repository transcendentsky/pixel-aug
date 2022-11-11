from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
# from scipy.misc import imread
import numpy as np
import cv2
from skimage.exposure import histogram
from scipy.stats import entropy
# from tutils import tfilename
from PIL import Image
# import torch
# import torchvision
import torchvision.transforms.functional as F
# from tutils import tfilename
from einops import rearrange, repeat
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')



def tmp_debug_hist():
    im = cv2.imread("D:\Documents\code\landmark/tproj/test/001.bmp", cv2.IMREAD_GRAYSCALE)
    # im = cv2.imread(tfilename('/home1/quanquan/datasets/Cephalometric/', "RawImage/TrainingData", "001.bmp"), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (384, 384))
    im_patch = im[50:114, 84:148]
    # im_patch = im[280:344, 268:332]
    # hist = im_to_hist(im)

    def _tmp_aug(patch):
        patch_aug = Image.fromarray(patch)
        patch_aug = F.adjust_brightness(patch_aug, 0.8)
        patch_aug = F.adjust_contrast(patch_aug, 1.5)
        patch_aug = np.array(patch_aug)
        return patch_aug

    def _tmp_crop_aug(im, loc, ps1=192, ps2=64):
        loc = [82, 116]


    def _tmp_get_entr_patch(im_patch):
        # normalize
        # import ipdb; ipdb.set_trace()
        im_patch = (im_patch - im_patch.mean()) / (im_patch.std()+1e-8)
        # import ipdb; ipdb.set_trace()
        fea = np.zeros((256,))
        hist, idx = histogram(im_patch, nbins=256)
        idx_int = (idx * 24).round().astype(int)
        # print(hist, idx, idx_int)
        # for hi, idi in zip(hist, idx):
        for i, (hi, idi_int) in enumerate(zip(hist, idx_int)):
            idi_int += 128
            if idi_int >= 256 or idi_int < 0:
                continue
            fea[idi_int] = hi
        entr = entropy(fea)
        return entr, fea, hist, idx, idx_int #idx*16

    im_patch_aug = _tmp_aug(im_patch)
    # entr1, fea1, hist, idx, idx_int = _tmp_get_entr_patch(im_patch); print(entr1)
    # entr2, fea2, hist2, idx2, idx_int2 = _tmp_get_entr_patch(im_patch_aug); print(entr2)


    def kde_estimate(hist, idx):
        x_test = np.linspace(-0, 2, 256)[:, np.newaxis]
        bandwidth = np.linspace(1, 9, 41) # [0.05,  .05, 1, 1.2, 1.5]
        # print("bandwidth: ", bandwidth)
        # kde_model = KernelDensity(kernel='gaussian')
        # grid = GridSearchCV(kde_model, {'bandwidth': bandwidth})
        # grid.fit(hist[:, np.newaxis], idx[:, np.newaxis])
        # print(hist.shape, idx.shape)
        # kde_model = grid.best_estimator_
        kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1)
        print("kde bandwith", kde_model.bandwidth)
        kde_model.fit(hist[:, np.newaxis], idx[:, np.newaxis])
        score = kde_model.score_samples(x_test)
        return score, x_test

    im_values = rearrange(im_patch, "h w -> (h w)")
    score, x_test = kde_estimate(np.ones_like(im_values), im_values)
    im_aug_values = rearrange(im_patch_aug, "h w -> (h w)")
    score2, x_test2 = kde_estimate(np.ones_like(im_aug_values), im_aug_values)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,8))
    plt.subplot(212)
    # print(x_test.shape, hist.shape, idx.shape, score.shape)
    plt.bar(x_test[:,0], np.exp(score), alpha=0.4)
    plt.bar(x_test[:,0], -np.exp(score2), alpha=0.4)
    plt.subplot(211)
    plt.bar(im_values, np.ones_like(im_values), alpha=0.4)
    plt.bar(im_aug_values, -np.ones_like(im_aug_values), alpha=0.4)
    plt.show()

    print(np.exp(score).shape)
    print(np.exp(score2).shape)
    mi = mutual_info_regression(np.exp(score)[:, np.newaxis], np.exp(score2))
    print("mi: ", mi)

    # entr2, fea2, hist, idx = _tmp_get_entr_patch(im_patch2); print(entr2)

    # draw_bar_auto_split(fea1, )
    fig, ax = plt.subplots(figsize=(28,8))
    ax.bar(im_values, np.ones_like(im_values), alpha=0.4)
    # plt.savefig("debug_entr_1.jpg")
    ax.bar(im_aug_values, -np.ones_like(im_aug_values), alpha=0.4)
    plt.savefig("debug_entr_2.jpg")
    plt.close()
    print("save fig")

    # print(entr1, entr2)
    loc1 = [352, 352]
    loc2 = [312, 300]

    # entr = hist_to_entropy(hist)
    # print(entr)
    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    tmp_debug_hist()