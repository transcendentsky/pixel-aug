# coding: utf-8
"""
    To prove that the variance exactly increased by entropy map
"""


import os.path
import random
import numpy as np
from sc.ssl2.ssl_probmap7 import *
from utils.tester.tester_gtv import Tester
from tutils import print_dict
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='ssl_probmap')
parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
# parser.add_argument('--func', default="test")
args = trans_args(parser)
logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)


def get_res3(pth, learner):
    tester = Tester(logger, config, retfunc=3, split='Train', cj_brightness=0.8, cj_contrast=0.6)
    learner.load(pth) if pth is not None else None
    learner.cuda()
    learner.eval()
    res = tester.test_func4(learner)
    return res


def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':
    reproducibility(0)
    fig = plt.figure()
    learner = Learner(logger, config)

    full_path = "/home1/quanquan/code/landmark/code/runs/ssl/ssl/baseline_ps64_3/ckpt/model_epoch_450.pth"
    res1 = get_res3(full_path, learner)
    print(res1)
    import ipdb; ipdb.set_trace()
    np.save(f"./tmp/tester_gtv3/_tmp_baseline450.npy", res1)
    exit()

    switch = False
    pths = []
    # pths += ["edge_inv"]
    # pths += ['gaussian_ps64_pseudo']
    pths += ['baseline_ps64_3']
    # pths += ["entr_ps64_inv2", ]
    # pths += ["prob_ks10", "entr_new_ushape"]
    length = np.array([1,1,1,1,1,1])
    for pth in pths:
        # epochs = [-1, 0,50,100,150,200,250] # 0, 50, 350, 400, ,300,350,400
        epochs = [450]
        for epoch in epochs:
            data_path = f"./tmp/tester_gtv3/ssl_{pth}_gtv1_epoch_{epoch}.npy"
            if switch or not os.path.exists(data_path):
                dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl/"
                # extname = "/ckpt_v/model_best.pth"
                extname = f'/ckpt/model_epoch_{epoch}.pth'
                full_path = dirname+pth+extname
                if epoch == -1:
                    full_path = None
                res1, res3, length = get_res3(full_path, learner)
                print(res1, res3)
                np.save(f"./tmp/tester_gtv3/ssl_{pth}_gtv1_epoch_{epoch}.npy", res1)
                np.save(f"./tmp/tester_gtv3/ssl_{pth}_gtv3_epoch_{epoch}.npy", res3)
                np.save(f"./tmp/tester_gtv3/ssl_{pth}_count.npy", length)
            else:
                res1 = np.load(data_path)
                # res3 = np.load(data_path[:-5] + "3.npy")
                print(res1)
            # import ipdb; ipdb.set_trace()
            plt.plot(np.arange(0,7), res1, label=f"{epoch} " + pth)
            # plt.plot(np.arange(0,7), res3, label="3 " + pth)

    plt.legend()
    plt.savefig("./tmp/tmp3.png")
    plt.savefig("./tmp/tmp3.pdf")

    # Use thres2_0, etc. models.
    pths = []
    # pths += ['thres2_0', 'thres2_3', 'thres2_4.5']
    learner = Learner(logger, config)
    length = np.array([1, 1, 1, 1, 1, 1])
    for pth in pths:
        # epochs = [-1, 0, 50, 100, 150, 200, 250]  # ,300,350,400
        # for epoch in epochs:
        data_path = f"./tmp/tester_gtv3/ssl_{pth}_gtv1.npy"
        if True or not os.path.exists(data_path):
            print("Processing ", data_path)
            dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap6/"
            # extname = "/ckpt_v/model_best.pth"
            extname = f'/ckpt_v/model_best.pth'
            full_path = dirname + pth + extname
            # if epoch == -1:
            #     full_path = None
            res1, res3, length = get_res3(full_path, learner)
            print(res1, res3)
            np.save(f"./tmp/tester_gtv3/ssl_{pth}_gtv1.npy", res1)
            np.save(f"./tmp/tester_gtv3/ssl_{pth}_gtv3.npy", res3)
            np.save(f"./tmp/tester_gtv3/ssl_{pth}_count.npy", length)
        else:
            res1 = np.load(data_path)
            # res3 = np.load(data_path[:-5] + "3.npy")
            print(res1)
        # import ipdb; ipdb.set_trace()
        plt.plot(np.arange(0, 7), res1, label=pth)
        # plt.plot(np.arange(0,7), res3, label="3 " + pth)

    plt.legend()
    plt.savefig("./tmp/tmp3.png")
    plt.savefig("./tmp/tmp3.pdf")
    exit(0)

