"""
    test InfoNCE loss changes
"""


import os.path
import random
import numpy as np
from sc.ssl2.ssl_probmap7 import *
from utils.tester.tester_gtv2 import Tester
from tutils import print_dict, tdir
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
    tester = Tester(logger, config, retfunc=1, split='Train')
    learner.load(pth) if pth is not None else None
    learner.cuda()
    learner.eval()
    res = tester.test(learner)
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
    tdir('./tmp/tester_gtv4/')
    fig = plt.figure()
    # plt.bar(np.arange(0,5), np.arange(0,5))
    # plt.savefig("./tmp/tmp.png")
    # exit()
    switch = True
    pths = []
    # pths += ["edge_inv"]
    # pths += ['gaussian_ps64_pseudo']
    pths += ['baseline_ps64_3']
    # pths += ["entr_ps64_inv2", ]
    # pths += ["prob_ks10", "entr_new_ushape"]
    learner = Learner(logger, config)
    length = np.array([1,1,1,1,1,1])
    for pth in pths:
        # epochs = [-1, 0,50,100,150,200,250] # 0, 50, 350, 400, ,300,350,400
        epochs = [450]
        for epoch in epochs:
            data_path = f"./tmp/tester_gtv4/ssl_{pth}_gtv_epoch_{epoch}.npy"
            if switch or not os.path.exists(data_path):
                dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl/"
                # extname = "/ckpt_v/model_best.pth"
                extname = f'/ckpt/model_epoch_{epoch}.pth'
                full_path = dirname+pth+extname
                if epoch == -1:
                    full_path = None
                res1, length = get_res3(full_path, learner)
                print(res1)
                np.save(f"./tmp/tester_gtv4/ssl_{pth}_gtv_epoch_{epoch}.npy", res1)
                np.save(f"./tmp/tester_gtv4/ssl_{pth}_count.npy", length)
            else:
                res1 = np.load(data_path)
                # res3 = np.load(data_path[:-5] + "3.npy")
                print(res1)
            # import ipdb; ipdb.set_trace()
            plt.plot(np.arange(0,7), res1, label=f"{epoch} " + pth)
            # plt.plot(np.arange(0,7), res3, label="3 " + pth)

    plt.legend()
    plt.savefig("./tmp/tester_gtv4.png")
    plt.savefig("./tmp/tester_gtv4.pdf")

    # Use thres2_0, etc. models.
    pths = []
    # pths += ['thres2_0', 'thres2_3', 'thres2_4.5']
    pths += ['gaussian_ps64_pseudo']
    pths += ["entr_t0.3_(p6)", ] # "entr_ps64_inv2",
    learner = Learner(logger, config)
    length = np.array([1, 1, 1, 1, 1, 1])
    for pth in pths:
        # epochs = [-1, 0, 50, 100, 150, 200, 250]  # ,300,350,400
        # for epoch in epochs:
        data_path = f"./tmp/tester_gtv4/ssl_{pth}_gtv.npy"
        if switch or not os.path.exists(data_path):
            print("Processing ", data_path)
            dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap7/"
            # extname = "/ckpt_v/model_best.pth"
            extname = f'/ckpt_v/model_latest.pth'
            full_path = dirname + pth + extname
            # if epoch == -1:
            #     full_path = None
            res1, length = get_res3(full_path, learner)
            print(res1)
            np.save(f"./tmp/tester_gtv4/ssl_{pth}_gtv.npy", res1)
            np.save(f"./tmp/tester_gtv4/ssl_{pth}_count.npy", length)
        else:
            res1 = np.load(data_path)
            # res3 = np.load(data_path[:-5] + "3.npy")
            print(res1)
        # import ipdb; ipdb.set_trace()
        plt.plot(np.arange(0, 7), res1, label=pth)
        # plt.plot(np.arange(0,7), res3, label="3 " + pth)

    plt.legend()
    plt.savefig("./tmp/tester_gtv4.png")
    plt.savefig("./tmp/tester_gtv4.pdf")
    exit(0)

