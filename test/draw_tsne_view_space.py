"""
    Positive pairs interpolation
"""

"""
    ssl_probmap3 + neg interpolate
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl_adapm import Cephalometric
from models.network_emb_study import UNet_Pretrained
from einops import rearrange, repeat
import random
import numpy as np
import torch.nn.functional as F
from skimage.exposure import histogram
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mutual_info_score
import matplotlib.patheffects as pe
import seaborn as sb
import os
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tutils import CSVLogger


torch.backends.cudnn.benchmark = True

def im_to_hist(im, patch_size=64, temp=None):
    assert im.shape == (384, 384), f"Got {im.shape}"
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


def patch_to_hist(patch, draw=False):
    x_test = np.linspace(-2, 2, 256)[:, np.newaxis]
    assert len(patch.shape) == 2, f"Got {patch.shape}"
    im_values = rearrange(patch, "h w -> (h w)")
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kde_model.fit(np.ones_like(im_values[:, np.newaxis]), im_values[:, np.newaxis])
    score = kde_model.score_samples(x_test)
    if draw:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.fill(x_test, np.exp(score))
        # plt.show()
        plt.savefig("debug_mi.png")
        import ipdb; ipdb.set_trace()
    return score


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


def cossim(fea1, fea2):
    cos_sim = F.cosine_similarity(torch.Tensor(fea1), torch.Tensor(fea2), dim=0)
    return cos_sim

def test(logger, config):
    import cv2
    import torchvision.transforms as transforms
    import PIL
    from PIL import Image
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    # ------------------------
    transform_list = [
        transforms.Resize((384, 384)),
        transforms.ColorJitter(brightness=1.6, contrast=1.2,
                               saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ]
    transform = transforms.Compose(transform_list)

    transform_list = [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ]
    transform2 = transforms.Compose(transform_list)

    # im_pth = cv2.imread("D:\Documents\code\landmark/tproj/test/001.bmp", cv2.IMREAD_GRAYSCALE)
    im_pth = '/home1/quanquan/datasets/Cephalometric/' + "RawImage/TrainingData/001.bmp"
    im = Image.open(im_pth).convert('RGB')
    lm = np.array([[82,116], [312, 300], [92, 232], [312, 100]])
    # im_patch = im[50:114, 84:144]
    # im_patch2 = im[280:344, 268:332]

    # -----  Initialize model ------
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id,
                    collect_sim=False, split="Test1+2", upsample="bilinear")
    model = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
    # pth = "/home1/quanquan/code/landmark/code/runs/ssl/ssl_pos_ip/debug/ckpt_v/model_best.pth"
    # pth = '/home1/quanquan/code/landmark/code/runs/ssl/ssl/collect_sim/ckpt_v/model_best.pth'
    # pth = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_mi/collect_sim/ckpt/best_model_epoch_30.pth'
    pth = '/home1/quanquan/code/landmark/code/runs/ssl/ssl/ps64/ckpt/model_epoch_150.pth'
    pretrained_dict = torch.load(pth)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model.eval()

    def concat_fea(fea_list, lm):
        feas = []
        for i in range(5):
            lm_layer = lm // 2**(5-i)
            afea = fea_list[i][0, :, lm_layer[0], lm_layer[1]]
            feas.append(afea)
        return torch.concat(feas)

    # # ----- estimation demo ----
    # output = model(im.unsqueeze(0))
    # fea1 = concat_fea(output, lm[0])
    # fea2 = concat_fea(output, lm[1])
    # import ipdb; ipdb.set_trace()

    # ----- estimation multiple times -----
    noise_list = []
    info_list = []
    noise_list2 = []
    info_list2 = []
    sim1 = []
    sim2 = []
    sim11 = []
    sim22 = []

    for i in range(100):
        print("processing ", i, end="\r")
        im2 = transform(im).cuda() if i > 0 else transform2(im).cuda()
        output = model(im2.unsqueeze(0))
        noise = concat_fea(output, lm[0]).detach().cpu().numpy()
        infof = concat_fea(output, lm[1]).detach().cpu().numpy()
        noise2 = concat_fea(output, lm[2]).detach().cpu().numpy()
        infof2 = concat_fea(output, lm[3]).detach().cpu().numpy()
        noise_list.append(noise)
        info_list.append(infof)
        noise_list2.append(noise2)
        info_list2.append(infof2)
        if i >0:
            sim1.append(cossim(noise, noise_list[0]))
            sim2.append(cossim(infof, info_list[0]))
            sim11.append(cossim(noise2, noise_list2[0]))
            sim22.append(cossim(infof2, info_list2[0]))

        # output2 = model(im2[:, 50:114, 84:148].unsqueeze(0))
        # output3 = model(im2[:, 280:344, 268:332].unsqueeze(0))
        # output22 = model(im2[:, 60:60+64, 200:200+64].unsqueeze(0))
        # output33 = model(im2[:, 280:344, 68:68+64].unsqueeze(0))
        # # print(output2[0].shape, output3[0].shape)
        # noise2 = concat_fea(output2, np.array([32, 32])).detach().cpu().numpy()
        # infof2 = concat_fea(output3, np.array([32, 32])).detach().cpu().numpy()
        # noise3 = concat_fea(output22, np.array([32, 32])).detach().cpu().numpy()
        # infof3 = concat_fea(output33, np.array([32, 32])).detach().cpu().numpy()
        # noise_list.append(noise2)
        # info_list.append(infof2)
        # noise_list2.append(noise3)
        # info_list2.append(infof3)
        # if i > 0:
        #     sim1.append(cossim(noise2, noise_list[0]))
        #     sim2.append(cossim(infof2, info_list[0]))
        #     sim11.append(cossim(noise3, noise_list2[0]))
        #     sim22.append(cossim(infof3, info_list2[0]))

    csvlogger = CSVLogger("./", name="_tmp_sim_record2.csv", mode="a+")
    res = {"noise1": np.mean(sim1), "info1": np.mean(sim2), "noise2": np.mean(sim11), "info2": np.mean(sim22), "pth": pth}
    csvlogger.record(res)
    print("Results: ", res)
    np.save("noise_list.npy", noise_list)
    np.save("info_list.npy", info_list)
    np.save("noise_list2.npy", noise_list2)
    np.save("info_list2.npy", info_list2)
    # import ipdb; ipdb.set_trace()

    draw_tsne(logger, config)


def draw_mi(logger, config, **kwargs):
    import torchvision.transforms as transforms
    from PIL import Image

    transform_list = [
        transforms.Resize((384, 384)),
        transforms.ColorJitter(brightness=1.6, contrast=1.2,
                               saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ]
    transform = transforms.Compose(transform_list)

    transform_list = [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ]
    transform2 = transforms.Compose(transform_list)

    im_pth = '/home1/quanquan/datasets/Cephalometric/' + "RawImage/TrainingData/001.bmp"
    im = Image.open(im_pth).convert('RGB')
    lm = np.array([[82,116], [312, 300]])
    noise_list = []
    info_list = []
    noise_list2 = []
    info_list2 = []
    sim1 = []
    sim2 = []
    sim11 = []
    sim22 = []

    for i in range(100):
        print("processing ", i, end="\r")
        im2 = transform(im) if i > 0 else transform2(im)
        im2 = im2[0].detach().numpy()
        assert len(im2.shape) == 2, f"Got {im2.shape}"
        noise2 = patch_to_hist(im2[ 50:114, 84:148])
        infof2 = patch_to_hist(im2[ 280:344, 268:332])
        noise3 = patch_to_hist(im2[ 60:60+64, 200:200+64])
        infof3 = patch_to_hist(im2[ 280:344, 68:68+64])
        # print(noise2.shape, noise3.shape)
        noise_list.append(noise2)
        info_list.append(infof2)
        noise_list2.append(noise3)
        info_list2.append(infof3)
        if i > 0:
            sim1.append(mutual_info_score(noise2, noise_list[0]))
            sim2.append(mutual_info_score(infof2, info_list[0]))
            sim11.append(mutual_info_score(noise3, noise_list2[0]))
            sim22.append(mutual_info_score(infof3, info_list2[0]))

    print("cos sim: ", np.mean(sim1), np.mean(sim2), np.mean(sim11), np.mean(sim22))
    np.save("noise_list.npy", noise_list)
    np.save("info_list.npy", info_list)
    np.save("noise_list2.npy", noise_list2)
    np.save("info_list2.npy", info_list2)
    draw_tsne(logger, config)


def draw_tsne(logger, config, **kwargs):
    from tutils.tutils.visualizers.tsne import TSNE
    noise_list = np.load("noise_list.npy")
    info_list = np.load("info_list.npy")
    noise_list2 = np.load("noise_list2.npy")
    info_list2 = np.load("info_list2.npy")
    x = np.concatenate([noise_list, noise_list2, info_list, info_list2], axis=0)
    y = np.ones((x.shape[0],))
    y[0:100] = 0
    y[100:200] = 1
    y[200:300] = 2
    y[300:400] = 3
    print(x.shape, y.shape)

    digits_final = TSNE(perplexity=30).fit_transform(x)
    #Play around with varying the parameters like perplexity, random_state to get different plots
    plot(digits_final,y)


def plot(x, colors, num_label=10, fname="display.png"):
    """
    With the above line, our job is done. But why did we even reduce the dimensions in the first place?
    To visualise it on a graph.
    So, here is a utility function that helps to do a scatter plot of thee transformed data
    """
    palette = np.array(sb.color_palette("hls", num_label))  #Choosing color palette

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    parent, tail = os.path.split(fname)
    ax = plt.subplot(aspect='equal')
    # ax.title.set_text(tail)
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    # Add the labels for each digit.
    plt.savefig(fname)
    txts = []
    for i in range(num_label):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    # return f, ax, txts
    fname2 = fname.replace(".png", "_with_id.png")
    parent, tail = os.path.split(fname2)
    # ax.title.set_text(tail)
    plt.savefig(fname2)
    plt.close()

if __name__ == '__main__':
    EX_CONFIG = {
        "base": {
            'base_dir': "/home1/quanquan/code/landmark/code/runs/draw/",
        },
        "special": {
            "cj_brightness": 1.6,  # 0.15
            "cj_contrast": 1.2,  # 0.25
            "cj_saturation": 0.3,  # 0.
            "cj_hue": 0.2,  # 0
            "weighted_t": 1 / 600,  # temperature
            "pretrain_model": "/home1/quanquan/code/landmark/code/runs/ssl/ssl_pos_ip/debug/ckpt/model_epoch_500.pth",
            "non_local": True,
        },
    }
    args = trans_args()
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    print_dict(config)
    # import ipdb; ipdb.set_trace()

    eval(args.func)(logger, config)
