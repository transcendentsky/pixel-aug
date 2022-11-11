from sklearn.metrics import mutual_info_score
# from scipy.misc import imread
import numpy as np
import cv2
from skimage.exposure import histogram
from scipy.stats import entropy
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as F
from tutils import tfilename
from einops import rearrange, repeat
from sklearn.neighbors import KernelDensity
import os
from sklearn.model_selection import GridSearchCV
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')


im = cv2.imread("001.bmp", cv2.IMREAD_GRAYSCALE)

def mutual_info_of_patches(im, patch_size=64, temp=None):
    assert len(im.shape) == 2, f"GOt {im.shape}"
    h, w = im.shape
    mi = np.zeros((h, w))
    ps_h = patch_size // 2
    for i in range(h-ps_h):
        for j in range(w-ps_h):
            l1 = max(0, i-ps_h)
            l2 = max(0, j-ps_h)
            patch = im[l1:i+ps_h, l2:j+ps_h]
            hist, idx = histogram(patch, nbins=256)
            fea = np.zeros((256,))
            for hi, idi in zip(hist, idx):
                fea[idi] = hi
            mii = mutual_info_score(temp, fea)
            mi[i, j] = mii
    return mi


def im_to_hist(im, patch_size=64, temp=None):
    # assert im.shape == (384, 384), f"Got {im.shape}"
    assert len(im.shape) == 2, f"GOt {im.shape}"
    h, w = im.shape
    # mi = np.zeros((h, w))
    ps_h = patch_size // 2
    fea_matrix = np.zeros((256, h, w))
    for i in range(h):
        for j in range(w):
            l1 = max(0, i-ps_h)
            l2 = max(0, j-ps_h)
            patch = im[l1:i+ps_h, l2:j+ps_h]
            hist, idx = histogram(patch, nbins=256)
            # fea = np.zeros((256,))
            for hi, idi in zip(hist, idx):
                # print(hi, idi, i, j)
                fea_matrix[idi, i, j] = hi
            # fea_matrix[:, i, j] /= fea_matrix[:, i, j].sum()
            # assert
            # entr = entropy(fea_matrix[:, i, j])
            # if np.isnan(entr) :
            #     import ipdb;ipdb.set_trace()
            # mii = mutual_info_score(temp, fea)
            # mi[i, j] = mii
    return fea_matrix


def hist_to_entropy(hist):
    c, h, w = hist.shape
    entr_map = np.zeros((h, w))
    for hi in range(h):
        for wi in range(w):
            fea = hist[:, hi, wi]
            # print(fea.shape)
            entr = entropy(fea)

            if np.isnan(entr) :
                import ipdb; ipdb.set_trace()
            entr_map[hi, wi] = entr
    return entr_map


def hist_to_mi(hist, temp):
    c, h, w = hist.shape
    entr_map = np.zeros((h, w))
    for hi in range(h):
        for wi in range(w):
            fea = hist[:, hi, wi]
            entr = mutual_info_score(fea, temp)
            if np.isnan(entr):
                import ipdb; ipdb.set_trace()
            entr_map[hi, wi] = entr
    return entr_map



def mutual_info_of_near_samples(hist, lm=None, ps=64):
    temp = hist[:, lm[0], lm[1]]
    # hist, idx = histogram(temp, nbins=256)
    # fea = np.zeros((256,))
    # for hi, idi in zip(hist, idx):
    #     fea[idi] = hi
    # print("temp fea: ", fea)

    # fea2 = np.zeros((256,))
    offsets = [-1, 1]
    for i in offsets:
        for j in offsets:
            fea = hist[:, lm[0]+i, lm[1]+j]
            mi = mutual_info_score(temp, fea)
            print(mi)
    import ipdb; ipdb.set_trace()

def test_near_mi():
    pass

def test_mi_of_aug_img():
    # lm0: min 0.7, max 1.75
    # im = cv2.imread("001.bmp", cv2.IMREAD_GRAYSCALE)
    im = cv2.imread(tfilename('/home1/quanquan/datasets/Cephalometric/', "RawImage/TrainingData", "001.bmp"), cv2.IMREAD_GRAYSCALE)
    entr = np.load('/home1/quanquan/datasets/Cephalometric/entr1/train/1.npy')
    im = cv2.resize(im, (384, 384))
    # lm = [165, 291]
    lm = [100,308]
    # print()
    ps_half = 32
    patch = im[lm[0]-ps_half:lm[0]+ps_half, lm[1]-ps_half:lm[1]+ps_half]
    

    def get_fea(patch):
        fea = np.zeros((256,))
        hist, idx = histogram(patch, nbins=256)
        for hi, idi in zip(hist, idx):
            # print(hi, idi, i, j)
            fea[idi] = hi
        return fea

    mean = np.mean(patch).astype(int)
    fea1 = get_fea(patch)
    cv2.imwrite("patch1.jpg", patch)

    # fn_aug = transforms.ColorJitter(brightness=0.9)
    patch_aug = Image.fromarray(patch)
    patch_aug = F.adjust_brightness(patch_aug, 1.7)
    patch_aug = F.adjust_contrast(patch_aug, 1.4)
    # patch_aug = F.adjust_saturation(patch_aug, 5)
    # patch_aug = F.adjust_hue(patch_aug, -0.2)
    patch_aug = np.array(patch_aug)
    # print("??????")
    cv2.imwrite("patch2.jpg", patch_aug)
    fea2 = get_fea(patch_aug)

    mi0 = mutual_info_score(fea1, fea1)
    mi = mutual_info_score(fea1, fea2)
    print("lm ", lm, ",  entr ", entr[lm[0], lm[1]], "self_mi, aug_mi ", mi0, mi)
    import ipdb; ipdb.set_trace()


def test_near():
    # im =
    # im = cv2.imread(tfilename('/home1/quanquan/datasets/Cephalometric/', "RawImage/TrainingData", "001.bmp"), cv2.IMREAD_GRAYSCALE)
    im = cv2.imread("001.bmp", cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (384, 384))
    hist = im_to_hist(im)
    print("-- get hist")
    entr = hist_to_entropy(hist)
    # entr = hist_to_mi(hist, hist[:, 300, 300])
    # mutual_info_of_near_samples(hist, [322, 311])
    # exit(0)
    print("-- get entr")
    entr = torch.Tensor(entr)
    entr /= entr.max()
    torchvision.utils.save_image(entr, "my_mi.jpg")
    # cv2.imwrite(entr)

    # import ipdb; ipdb.set_trace()


def draw_all_mi_maps():
    from datasets.ceph.ceph_test import Test_Cephalometric
    print("draw_all_mi_maps")
    im_dir = '/home1/quanquan/datasets/Cephalometric/RawImage/TrainingData'
    testset = Test_Cephalometric(pathDataset='/home1/quanquan/datasets/Cephalometric/', mode="Train")
    for i in range(len(testset)):
        fname = tfilename(f"tmp_visual/mi2/{i:03d}")
        if os.path.exists(fname + ".npy"):
            print("ignore ", fname + ".npy")
            continue
        data = testset.return_func_pure(i)
        # {"img": item['image'], "landmark_list": landmark_list, "name": item['ID'] + '.bmp'}
        # im = data['img']
        landmark_list = data['landmark_list']
        name = data['name']
        im = cv2.imread(tfilename(im_dir, name), cv2.IMREAD_GRAYSCALE)
        hist = im_to_hist(im, patch_size=10)
        print("Image processing Get hist ", i)
        entr_list = []
        for lm in landmark_list:
            print("landmark processing: ", lm)
            entr = hist_to_mi(hist, hist[:, lm[1], lm[0]])
            # entr = np.random.random((384,384))
            # assert entr.shape == (384,384), f"Got {entr.shape}"
            entr_list.append(entr)
        max_entr = np.stack(entr_list, 0).max(axis=0)
        # assert max_entr.shape == (384,384), f"Got {max_entr.shape}"

        np.save(fname + ".npy", max_entr)
        entr = torch.Tensor(max_entr)
        entr /= entr.max()
        torchvision.utils.save_image(entr, fname + ".jpg")

        # import ipdb; ipdb.set_trace()


def tmp_debug_hist():
    im = cv2.imread("D:\Documents\code\landmark/tproj/test/001.bmp", cv2.IMREAD_GRAYSCALE)
    # im = cv2.imread(tfilename('/home1/quanquan/datasets/Cephalometric/', "RawImage/TrainingData", "001.bmp"), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (384, 384))
    cv2.imwrite("ori_001.png", im)
    # im_patch = im[50:114, 84:148]
    im_patch = im[60:60+64, 200:200+64]
    cv2.imwrite("patch1.png", im_patch)
    im_patch2 = im[280:344, 68:68+64] # 268:332
    cv2.imwrite("patch2.png", im_patch2)
    # import ipdb; ipdb.set_trace()
    # hist = im_to_hist(im)

    def _tmp_aug(patch):
        patch_aug = Image.fromarray(patch)
        patch_aug = F.adjust_brightness(patch_aug, 0.8)
        patch_aug = F.adjust_contrast(patch_aug, 1.5)
        patch_aug = np.array(patch_aug)
        return patch_aug

    def _tmp_get_entr_patch(im_patch):
        # normalize
        # import ipdb; ipdb.set_trace()
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
    im_patch = (im_patch - im_patch.mean()) / (im_patch.std()+1e-8)
    entr1, fea1, hist, idx, idx_int = _tmp_get_entr_patch(im_patch); print(entr1)
    entr2, fea2, hist2, idx2, idx_int2 = _tmp_get_entr_patch(im_patch_aug); print(entr2)

    im_values = rearrange(im_patch, "h w -> (h w)")
    x_test = np.linspace(-4, 4, 256)[:, np.newaxis]
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1)
    print(hist.shape, idx.shape)
    # kde_model.fit(hist[:, np.newaxis], idx[:, np.newaxis])
    kde_model.fit(np.ones_like(im_values[:, np.newaxis]), im_values[:, np.newaxis])
    score = kde_model.score_samples(x_test)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.fill(x_test, np.exp(score))
    plt.show()

    mi = mutual_info_score(fea1, fea2)
    print("mi: ", mi)

    # entr2, fea2, hist, idx = _tmp_get_entr_patch(im_patch2); print(entr2)

    # draw_bar_auto_split(fea1, )
    fig, ax = plt.subplots(figsize=(28,8))
    ax.bar(idx, hist, alpha=0.4)
    # plt.savefig("debug_entr_1.jpg")
    ax.bar(idx2, -hist2, alpha=0.4)
    plt.savefig("debug_entr_2.jpg")
    plt.close()
    print("save fig")

    # print(entr1, entr2)
    loc1 = [352, 352]
    loc2 = [312, 300]

    # entr = hist_to_entropy(hist)
    # print(entr)
    # import ipdb; ipdb.set_trace()


def tmp_debug_views_by_model():
    import argparse
    from tutils import trans_args, trans_init
    from models.network_emb_study import UNet_Pretrained
    EX_CONFIG = {
        "special": {
            "cj_brightness": 1.6,  # 0.15
            "cj_contrast": 1.2,  # 0.25
            "cj_saturation": 0.,  # 0.
            "cj_hue": 0.,  # 0
            "weighted_t": 1 / 600,  # temperature
            "pretrain_model": "/home1/quanquan/code/landmark/code/runs/ssl/ssl_pos_ip/debug/ckpt/model_epoch_200.pth"
        },
        "training": {
            "load_pretrain_model": True,
            "lr": 0.0001,
        }
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
    state_dict = torch.load(config['special']['pretrain_model'])
    net.load_state_dict(state_dict)
    net.cuda()
    net.eval()
    norm = torchvision.transforms.Normalize([0], [1])
    # start views operations
    im = cv2.imread(tfilename('/home1/quanquan/datasets/Cephalometric/', "RawImage/TrainingData", "001.bmp"), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (384, 384))
    # im_patch = im[50:114, 84:144]
    # im_patch = im[280:344, 268:332]

    def _tmp_aug(patch):
        patch_aug = Image.fromarray(patch)
        patch_aug = F.adjust_brightness(patch_aug, 1.6)
        patch_aug = F.adjust_contrast(patch_aug, 1.2)
        patch_aug = np.array(patch_aug)
        return patch_aug

    im_aug = _tmp_aug(im)
    im = np.stack([im, im_aug], axis=0)
    im = repeat(im, "b h w -> b c h w", c=3)
    tensor = torch.Tensor(im).cuda()
    fea_matrix = net(norm(tensor))
    # import ipdb; ipdb.set_trace()

    def _tmp_sim(fea_matrix):
        # loc = np.array([82, 116])
        loc = np.array([312, 300])
        b, c, h, w = fea_matrix.shape
        loc = loc // (384//h)
        f1 = fea_matrix[0, :, loc[0], loc[1]]
        f2 = fea_matrix[1, :, loc[0], loc[1]]
        cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
        cos_sim = cosfn(rearrange(f1, "c -> 1 c 1 1"), rearrange(f2, "c -> 1 c 1 1"))
        return cos_sim

    total_sim = 1.0
    for i in range(5):
        sim = _tmp_sim(fea_matrix[i]).squeeze().cpu().item()
        total_sim *= sim
        print(sim, total_sim)
    print("total sim", total_sim)
    print(config['special']['pretrain_model'])
    # import ipdb; ipdb.set_trace()


def draw_bar(labels, values, fname="tbar.pdf", title=None, color="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    # fig = plt.figure(figsize=(11,6))
    fig, ax = plt.subplots(figsize=(14,8))
    if title is not None:
        fig.suptitle(title)
    # ax = fig.add_axes([0,0,1,1])
    assert len(labels) == len(values) + 1
    x_pos = [i for i, _ in enumerate(labels)]
    x_pos2 = np.array(x_pos[:-1])
    width = 0.5
    print(x_pos2)
    # import ipdb; ipdb.set_trace()
    ax.bar(x_pos2 + width, values, alpha=0.7, color=color)
    ax.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.5)

    fontsize_ticks = 22
    fontsize_label = 28
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    plt.xticks(x_pos[:-1], labels[:-1], fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig(fname)
    plt.close()
    print("Drawed img: ", fname)


if __name__ == '__main__':
    test_mi_of_aug_img()
    # test_near()
    # draw_all_mi_maps()
    # tmp_debug_hist()
    # tmp_debug_views_by_model()
