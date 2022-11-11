import cv2
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
# from scipy.misc import imread
import numpy as np
import cv2
from skimage.exposure import histogram
from scipy.stats import entropy
import matplotlib
from tutils import tfilename
from PIL import Image
import torchvision.transforms.functional as F
from scipy.stats import norm
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from skimage.filters.rank import entropy
# from scipy.stats import entropy
from skimage.morphology import disk
from tqdm import tqdm

"""
    Mutual info:
    0.8334361740522674 0.236187758371206

    Mutual info (self):
    2.781115538153201 0.6866188393380371

 
     Entropy:
    4.592848378070493 0.5059876207290258

 """

mi_mean = [0.58855864,0.9689842,0.88423108,0.41595766,0.93467499,0.87520715
            ,0.75355031,0.69097756,0.70706193,0.9617875,0.95781572,0.96094506
            ,0.98639255,0.89073955,0.96499311,0.71407015,0.93329412,0.91891464
            ,0.7271314]
mi_std = [0.26388173,0.12006105,0.12696776,0.31245908,0.10358765,0.20422783
            ,0.21599622,0.18820669,0.19157507,0.13725611,0.11465392,0.12241823
            ,0.11602296,0.19213057,0.09218982,0.18752875,0.1210562, 0.1017045
            ,0.28287337]

miself_mean = [2.51138927,3.18064627,3.1936682,2.11334156,3.26182041,2.72902131
            ,2.16312098,1.94328915,2.01532239,3.26112403,3.20813622,3.16552408
            ,3.12697808,2.64613941,3.25124197,1.97482131,3.206975,3.22484813
            ,2.66378744]
miself_std = [0.46076691,0.30574453,0.2845213,0.48100281,0.24713011,0.75816679
,0.75060892,0.65067037,0.6793318, 0.23704422,0.36165119,0.38236706
,0.30267382,0.60297583,0.24602729,0.60459519,0.21163927,0.24409171
,0.47145102]

entr_mean = [4.32922599,4.94955578,4.9402223,4.13369406,5.02937445,4.46131056
            ,4.12397278,4.07084514,4.08404715,4.8864486, 4.87044201,4.79871701
            ,4.7067016, 4.29701387,4.94452249,4.05091702,5.03718015,5.01795204
            ,4.53197619]
entr_std = [0.32104817,0.36015503,0.25001072,0.34826474,0.22153489,0.53958994
,0.44612867,0.39377246,0.4095514, 0.27827981,0.33560316,0.348146
,0.33046936,0.41912717,0.26862784,0.38550231,0.17205685,0.20877866
,0.32103766]

mre = [1.399, 1.5526,1.7562,2.9915 , 1.7145, 1.9027,
       1.8260, 1.3118, 1.1585, 4.4261, 2.2153, 2.8174,
       1.5348, 1.9849, 1.3075, 1.8713,1.5415, 3.7358,
       2.3073]
mre2 =[1.6767,1.7246,2.2157,2.5067,1.9819,2.2091,
       1.7173,1.3327,1.6392,2.2740,1.6451,1.7791,
       1.5442,1.7209,1.8208,2.9706,1.8948,1.9768,
       2.1830]

def get_fea(patch):
    fea = np.zeros((256,))
    hist, idx = histogram(patch, nbins=256)
    for hi, idi in zip(hist, idx):
        # print(hi, idi, i, j)
        fea[idi] = hi
    return fea


"""
    115.bmp: lm 0: 2.17 2.56, 0.52 0.36     |  0.588 2.511 4.321
            lm 3:  1.84 2.12, 0.55 0.40     |  0.415 2.113 4.133
            lm 7:  2.11 2.48, 0.49 0.32     |  0.690 1.943 4.070
            lm 17: 1.72 2.00, 0.40 0.20     |  0.918 3.224 5.017
            
    # averaged
    115.bmp: lm 0: 2.00 2.34, 0.55 0.40     |  0.588 2.511 4.321
            lm 3:  2.55 3.06, 0.47 0.29     |  0.415 2.113 4.133
            lm 7:  2.09 2.45, 0.46 0.29     |  0.690 1.943 4.070
            lm 17: 1.62 1.83, 0.54 0.40     |  0.918 3.224 5.017
"""

ALPHA = 0.3

def _tmp_fn(landmark_id):
    """ crop patch and augment """
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    pth = '/home1/quanquan/datasets/Cephalometric/'
    testset = Test_Cephalometric(pth, mode="Train")
    # Process: lm 0, 3, 7
    lms = testset.ref_landmarks(114)

    dirname = '/home1/quanquan/datasets/Cephalometric/RawImage/TrainingData/'

    params_list = []
    for i in range(150):
        i += 1
        # i = 115
        # landmark_id = 3

        im_name = f'{i:03d}.bmp'
        im = cv2.imread(dirname + im_name, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (384, 384))

        lm = lms[landmark_id] # [168, 163]
        # print("lm : ", lm)
        ps_half = 32
        patch = im[lm[1]-ps_half:lm[1]+ps_half, lm[0]-ps_half:lm[0]+ps_half]
        fea1 = get_fea(patch)
        # cv2.imwrite("patch1.jpg", patch)
        # fn_aug = transforms.ColorJitter(brightness=0.9)
        patch = Image.fromarray(patch)

        # print("=============================")
        cj_brightness = 1.
        cj_contrast = 1.
        params1 = [0,0]
        for i in range(100):
            patch_aug = F.adjust_brightness(patch, cj_brightness)
            patch_aug = F.adjust_contrast(patch_aug, cj_contrast)
            patch_aug = np.array(patch_aug)
            # cv2.imwrite("patch2.jpg", patch_aug)
            fea2 = get_fea(patch_aug)

            # mi0 = mutual_info_score(fea1.copy(), fea1.copy())
            mi = mutual_info_score(fea1, fea2)
            # print(f"entr: {entr_mean[landmark_id]}, br: {cj_brightness}, ct: {cj_contrast}, mi:", mi, mi_mean[landmark_id] - mi_std[landmark_id]) # mi0
            cj_brightness += 0.03
            cj_contrast += 0.04
            if mi < mi_mean[landmark_id] - mi_std[landmark_id]:
                break
            params1 = [cj_brightness, cj_contrast]
        # print("=============================")

        cj_brightness = 1
        cj_contrast = 1
        params2 = [0,0]
        for i in range(100):
            patch_aug = F.adjust_brightness(patch, cj_brightness)
            patch_aug = F.adjust_contrast(patch_aug, cj_contrast)
            patch_aug = np.array(patch_aug)
            # cv2.imwrite("patch2.jpg", patch_aug)
            fea2 = get_fea(patch_aug)

            # mi0 = mutual_info_score(fea1.copy(), fea1.copy())
            mi = mutual_info_score(fea1, fea2)
            # print(f"entr: {entr_mean[landmark_id]}, br: {cj_brightness}, ct: {cj_contrast}, mi:", mi, mi_mean[landmark_id] - mi_std[landmark_id])  # mi0
            cj_brightness -= 0.03
            cj_contrast -= 0.04
            if mi < mi_mean[landmark_id] - mi_std[landmark_id]:
                break
            params2 = [cj_brightness, cj_contrast]
        # print("=============================")
        params_list.append(params1 + params2)
        # import ipdb; ipdb.set_trace()

    params_list = np.array(params_list)
    mean_values = params_list.mean(axis=0)
    return mean_values
    # print(f"mean values ", mean_values)
    # import ipdb; ipdb.set_trace()


def _tmp_fn2(landmark_id, alpha):
    """ augment and crop patch (different from _tmp_fn() ) """
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    pth = '/home1/quanquan/datasets/Cephalometric/'
    testset = Test_Cephalometric(pth, mode="Train")
    # Process: lm 0, 3, 7
    lms = testset.ref_landmarks(114)

    dirname = '/home1/quanquan/datasets/Cephalometric/RawImage/TrainingData/'

    params_list = []
    for i in range(150):
        i += 1
        # i = 115
        # landmark_id = 3

        im_name = f'{i:03d}.bmp'
        im = cv2.imread(dirname + im_name, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (384, 384))

        lm = lms[landmark_id]  # [168, 163]
        ps_half = 32
        patch = im[lm[1] - ps_half:lm[1] + ps_half, lm[0] - ps_half:lm[0] + ps_half]
        fea1 = get_fea(patch)
        im = Image.fromarray(im)

        # print("=============================")
        cj_brightness = 1.
        cj_contrast = 1.
        params1 = [0, 0]
        for i in range(100):
            im_aug = F.adjust_brightness(im, cj_brightness)
            im_aug = F.adjust_contrast(im_aug, cj_contrast)
            im_aug = np.array(im_aug)
            # cv2.imwrite("patch2.jpg", patch_aug)
            patch_aug = im_aug[lm[1] - ps_half:lm[1] + ps_half, lm[0] - ps_half:lm[0] + ps_half]
            fea2 = get_fea(patch_aug)
            # mi0 = mutual_info_score(fea1.copy(), fea1.copy())
            mi = mutual_info_score(fea1, fea2)
            # print(f"entr: {entr_mean[landmark_id]}, br: {cj_brightness}, ct: {cj_contrast}, mi:", mi, mi_mean[landmark_id] - mi_std[landmark_id]) # mi0
            cj_brightness += 0.03
            cj_contrast += 0.04
            if mi < mi_mean[landmark_id] * alpha:  # - mi_std[landmark_id]:
                break
            params1 = [cj_brightness, cj_contrast]
        # print("=============================")

        cj_brightness = 1
        cj_contrast = 1
        params2 = [0, 0]
        for i in range(100):
            im_aug = F.adjust_brightness(im, cj_brightness)
            im_aug = F.adjust_contrast(im_aug, cj_contrast)
            im_aug = np.array(im_aug)
            # cv2.imwrite("patch2.jpg", patch_aug)
            patch_aug = im_aug[lm[1] - ps_half:lm[1] + ps_half, lm[0] - ps_half:lm[0] + ps_half]
            fea2 = get_fea(patch_aug)

            # mi0 = mutual_info_score(fea1.copy(), fea1.copy())
            mi = mutual_info_score(fea1, fea2)
            # print(f"entr: {entr_mean[landmark_id]}, br: {cj_brightness}, ct: {cj_contrast}, mi:", mi, mi_mean[landmark_id] - mi_std[landmark_id])  # mi0
            cj_brightness -= 0.03
            cj_contrast -= 0.04
            if mi < mi_mean[landmark_id] * alpha:  # - mi_std[landmark_id]:
                break
            params2 = [cj_brightness, cj_contrast]
        # print("=============================")
        params_list.append(params1 + params2)
        # import ipdb; ipdb.set_trace()

    params_list = np.array(params_list)
    mean_values = params_list.mean(axis=0)
    return mean_values


def get_entr():
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    pth = '/home1/quanquan/datasets/Cephalometric/'
    testset = Test_Cephalometric(pth, mode="Train")
    # Process: lm 0, 3, 7
    lms_list = []
    for i in range(150):
        lms_list.append(testset.ref_landmarks(i))
    lms_list = np.array(lms_list) # (150, 19, 2)
    print(lms_list.shape)

    entr_list= []
    dirname = '/home1/quanquan/datasets/Cephalometric/RawImage/TrainingData/'
    for i in tqdm(range(150)):
        im_name = f'{i+1:03d}.bmp'
        im = cv2.imread(dirname + im_name, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (384, 384))
        entr_img = entropy(im, disk(10))
        entr_list.append([entr_img[loc[1],loc[0]] for loc in lms_list[i]])
    entr_list = np.array(entr_list)
    print(entr_list.shape)
    np.save('./cache/entr_values_384.npy',entr_list)
    import ipdb; ipdb.set_trace()


def draw_line():
    """ please refer to [Notebook] search_optimal_ceph.ipynb """
    import pandas as pd
    data = np.load("./cache/optimal_brct.npy", allow_pickle=True)
    data = data.tolist()
    df = pd.DataFrame.from_dict(data)

    fig = sns.lineplot(x="entr_mean", y="cj_br_ceil", data=df)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig("./tmp/_search_optimal_ceph.png")
    import ipdb; ipdb.set_trace()


def process():
    new_entr = np.load('./cache/entr_values_384.npy')
    new_entr = new_entr.mean(axis=0)
    print(new_entr)
    res_list_med = []
    res_list_high = []

    for i in tqdm(range(19)):
        params = _tmp_fn2(i, alpha=ALPHA)
        res_d = {
            "index": i,
            "mi_mean": mi_mean[i],
            "mi_std": mi_std[i],
            "miself_mean": miself_mean[i],
            "miself_std": miself_std[i],
            "entr_mean": entr_mean[i],
            "entr_std": entr_std[i],
            "cj_br_ceil": params[0],
            "cj_ct_ceil": params[1],
            "cj_br_floor": params[2],
            "cj_ct_floor": params[3],
            "new_entr": new_entr[i],
            'mre_lm': mre[i],
            'mre_lm2': mre2[i],
        }
        
        if entr_mean[i] < 4.5:
            res_list_med.append(res_d)
        elif entr_mean[i] >= 4.5:
            res_list_high.append(res_d)
        # print(res_d)
    np.save(f"./cache/optimal_brct_im2_alpha{ALPHA}_med.npy", res_list_med)
    np.save(f"./cache/optimal_brct_im2_alpha{ALPHA}_high.npy", res_list_high)
    return res_list_med, res_list_high

def _tmp_process():
    data = np.load("./cache/optimal_brct.npy", allow_pickle=True).tolist()
    new_entr = np.load('./cache/entr_values_384.npy')
    new_entr = new_entr.mean(axis=0)
    print(new_entr)
    for i, datai in enumerate(data):
        datai["new_entr"] = new_entr[i]
        datai['mre_lm'] = mre[i]
        datai['mre_lm2'] = mre2[i]
        print(datai)
    np.save("./cache/optimal_brct.npy", data)

if __name__ == '__main__':
    res1, res2 = process()
    ares1 = np.array([r['cj_br_ceil'] for r in res1])
    ares2 = np.array([r['cj_br_ceil'] for r in res2])
    print(ALPHA, ares1.mean())
    print(ALPHA, ares2.mean())

    # _tmp_process()
    # draw_line()
    # get_entr()