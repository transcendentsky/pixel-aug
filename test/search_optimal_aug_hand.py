import cv2
from sklearn.metrics import mutual_info_score
# from scipy.misc import imread
import numpy as np
import cv2
from skimage.exposure import histogram
from scipy.stats import entropy
import matplotlib
from tutils import tfilename
from scipy.stats import norm
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')



def draw_bar_auto_split(values, tag=0, draw_guassian_line=False):
    thresholds = np.arange(0., 3.6, 0.2).astype(float)
    print()
    thresholds[0] = 0
    thresholds[-1] = 999
    # print(thresholds, thresholds.shape)
    mre = np.array(values)
    print(mre.shape)
    # print(mre)
    length_collect = []
    pre_len = 0
    for i in range(len(thresholds)-1):
        ind = np.where(mre<=thresholds[i+1] )[0]
        length_collect.append(len(ind) - pre_len)
        pre_len = len(ind)
        # print("mmm: ", len(ind), length_collect)
    length_collect = np.array(length_collect) / len(mre)
    thresholds_str = [f"{i:.2f}" for i in thresholds]
    print(thresholds_str)

    x_test = None
    y = None
    if draw_guassian_line:
        mean = mre.mean()
        std = mre.std()
        x_test = np.linspace(0, 3, 100)
        print("????", mean, std)
        def _gaussian(x, mean, std):
            a = 1 / np.sqrt(2 * 3.141592 * std ** 2)
            y = a * np.exp(-(x-mean)**2 / (2 * std**2))
            return y
        y = [_gaussian(x, mean, std) for x in x_test]
        y = np.array(y)
        print(y)
        print(x_test)
        # plt.plot(x, y)
    draw_bar(thresholds_str, length_collect, fname=f"hand_mi_lm{tag}.png", color="blue", xlabel="Mutual Information (MI)", ylabel="Percentage (%)", ex_x=x_test, ex_y=y)

    # draw_bar(thresholds, length_collect, fname=f"tbar_mre_n1.pdf", color="blue", xlabel="Mean Radial Error (MRE)", ylabel="Quantity (%)")
    return thresholds_str, length_collect


def draw_bar(labels, values, fname="tbar.pdf", title=None, color="red", set_font=None, xlabel="x", ylabel="y", ex_x=None, ex_y=None):
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

    if ex_x is not None and ex_y is not None:
        ax.plot(ex_x, ex_y, color="green", label="gaussian")
    plt.savefig(fname)
    plt.close()
    print("Drawed img: ", fname)


EX_CONFIG = {
    'dataset': {
        'name': 'Cephalometric',
        'pth': '/home1/quanquan/datasets/hand/hand/',
        # 'entr': '/home1/quanquan/datasets/Cephalometric/entr1/train/',
        # 'prob': '/home1/quanquan/datasets/Cephalometric/prob/train/',
        'n_cls': 37,
    }
}

def test_all(landmark_list, landmark_id=0):
    from torchvision import transforms
    from PIL import Image
    from tutils import tfilename

    pth = EX_CONFIG['dataset']['pth']
    # if landmark_list is None:
    from datasets.hand.hand_basic import HandXray
    testset = HandXray(pth, split="Train", ret_mode="no_process")
    # item, landmark_list, template_patches = testset.__getitem__(0)
    # data = testset.__getitem__(id_oneshot)
    landmark_list = []
    img_list = []
    for i in range(len(testset)):
        # print("i ", i)
        item, landmarks, index, img_shape, ori_img = testset._get_data(i, return_ori=True)
        landmark = landmarks[landmark_id]
        landmark_list.append(landmark)
        img_list.append(ori_img)

    def get_fea(patch):
        fea = np.zeros((256,))
        hist, idx = histogram(patch, nbins=256)
        for hi, idi in zip(hist, idx):
            # print(hi, idi, i, j)
            fea[idi] = hi
        return fea

    im_list = []
    fea_list = []
    # landmark_id = 3
    for i in range(len(landmark_list)):
        im = np.array(img_list[i])
        im = cv2.resize(im, (384, 384))
        lm = landmark_list[i]
        ps_half = 32
        patch = im[max(lm[0]-ps_half, 0):lm[0]+ps_half, max(lm[1]-ps_half, 0):lm[1]+ps_half]
        if not (patch.shape[0] > 0 and patch.shape[1] > 0):
            import ipdb; ipdb.set_trace()
        # assert patch.shape == (64, 64), f"Got {patch.shape}"
        fea1 = get_fea(patch)
        fea_list.append(fea1)

    fea0 = fea_list[0]
    # mi0 = mutual_info_score(fea0, fea0)
    mi_list = []
    for i in range(len(landmark_list)):
        mii = mutual_info_score(fea0, fea_list[i])
        mi_list.append(mii)
    # print(mi_list)
    mi_list = np.array(mi_list)
    print("max mi: ", mi_list.max())
    print("min mi: ",  mi_list.min())
    # import ipdb; ipdb.set_trace()
    # draw_bar_auto_split(mi_list)
    # return mi_list.max(), mi_list.min(), mi_list.mean()
    return mi_list


def ana_test_all():
    res_list = []
    for i in range(37):
        res = test_all(None, i)
        print("---------------------------")
        print(res)
        print("---------------------------")
        res_list.append(res)
        # import ipdb; ipdb.set_trace()
    res_total = np.concatenate(res_list, axis=0)
    draw_bar_auto_split(res_total, tag=99)
    np.save(tfilename("./cache/res_total_lm_hand.npy"), res_total)


if __name__ == '__main__':
    data = np.load(tfilename("./cache/res_total_lm_hand.npy"))
    import ipdb; ipdb.set_trace()
    # ana_test_all()