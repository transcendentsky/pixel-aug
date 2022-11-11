"""
    Search optimal params for Augmentation / entropy temperature


"""

"""
    This script is for adjusting ColorJitter's params,
        brightness , contrast, saturation, hue

    base code copied from ssl_hand_probmap.py (ps 64

    v0.2 : + (probmap + entr_map)
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
import numpy as np
from einops import rearrange


torch.backends.cudnn.benchmark = True


EX_CONFIG = {
    "special": {
        "cj_brightness": 0, # 0.15
        "cj_contrast": 0., # 0.25
        "pretrained_model": '/home1/quanquan/code/landmark/code/runs/ssl/ssl_pos_ip/debug/ckpt_v/model_best.pth',
    }
}


cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product(feature, template):
    cos_sim = cosfn(rearrange(template, "c -> 1 c 1 1"), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def test(logger, config):
    from datasets.ceph.ceph_test import Test_Cephalometric
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")

    net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
    net.load_state_dict(torch.load(config['special']['pretrained_model']))
    net.cuda()
    id_oneshot = 114

    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train")
    # landmark_list_list = []
    # for i in range(len(testset)):
    #     landmark_list = testset.ref_landmarks(i)
    #     landmark_list_list.append(landmark_list)
    # config_spe = config['special']
    # dataset_train = Cephalometric(config['dataset']['pth'],
    #                               patch_size=config['special']['patch_size'],
    #                               entr_map_dir=config['dataset']['entr'],
    #                               prob_map_dir=config['dataset']['prob'],
    #                               mode="Train", use_prob=True, pre_crop=False, retfunc=2,
    #                               cj_brightness=config_spe['cj_brightness'],
    #                               cj_contrast=config_spe['cj_contrast'])
    # dataset_train.prob_map_for_all(landmark_list_list)

    # ----------------
    # index = np.random.randint(0, len(dataset_train))
    data1 = testset.__getitem__(114)
    img = data1['img'].cuda()
    landmark_list = data1['landmark_list']
    raw_fea_list1 = net(img.unsqueeze(0))

    s_list = []
    for i in range(len(testset)):
        print("Processing ", i, end="\r")
        # index = np.random.randint(0, len(dataset_train))
        index = i
        data2 = testset.__getitem__(index)

        # raw_imgs1 = data1['raw_imgs'].cuda()
        # raw_loc1  = data1['raw_loc'].cuda()
        raw_imgs2 = data2['img'].cuda()

        raw_fea_list2 = net(raw_imgs2.unsqueeze(0))

        m_list = []
        for m in range(19):
            n_list = []
            for n in range(len(raw_fea_list2)):
                scale = 2 ** (5 - n)
                h, w = landmark_list[m][0] // scale, landmark_list[m][1] // scale
                fea1 = raw_fea_list1[n][0, :, h, w]
                fea2 = raw_fea_list2[n]
                s = match_inner_product(fea2, fea1)
                v = s.max().item()
                n_list.append(v)
            m_list.append(n_list)
        s_list.append(m_list)

    s_list = np.array(s_list)
    print(s_list.mean(axis=0))
    np.save("_tmp_s_list.npy", s_list)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    print_dict(config)
    eval(args.func)(logger, config)

