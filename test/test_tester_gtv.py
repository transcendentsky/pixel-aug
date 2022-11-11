"""
    Test all pixels
"""
import os.path

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


def get_res(pth, learner):
    tester = Tester(logger, config, retfunc=1, split='subtrain')
    learner.load(pth)
    learner.cuda()
    learner.eval()
    res = tester.test(learner)
    return res


if __name__ == '__main__':
    fig = plt.figure()
    # plt.bar(np.arange(0,5), np.arange(0,5))
    # plt.savefig("./tmp/tmp.png")
    # exit()
    switch = False
    pths = []
    # pths += ["edge_inv"]
    # pths += ['gaussian_ps64_pseudo']
    pths += [ "entr_t0.3_(p6)", ] # "entr_ps64_inv2",
    # pths += ["prob_ks10", "entr_new_ushape"]
    learner = Learner(logger, config)
    length = np.array([1,1,1,1,1,1])
    for pth in pths:
        data_path = f"./tmp/tester_gtv/ssl_probmap7_{pth}_gtv.npy"
        if switch or not os.path.exists(data_path):
            dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap7/"
            extname = "/ckpt_v/model_best.pth"
            full_path = dirname+pth+extname
            res, length = get_res(full_path, learner)
            print(res)
            np.save(f"./tmp/tester_gtv/ssl_probmap7_{pth}_gtv.npy", res)
            np.save(f"./tmp/tester_gtv/ssl_probmap7_{pth}_count.npy", length)
        else:
            res = np.load(data_path)
            print(res)
        # import ipdb; ipdb.set_trace()
        plt.plot(np.arange(0,7), res, label=pth)

    plt.legend()
    plt.savefig("./tmp/tmp.png")
    plt.savefig("./tmp/tmp.pdf")
    # exit()

    # ssl
    pths = []
    pths += ["baseline_ps64_3"]
    for pth in pths:
        data_path = f"./tmp/tester_gtv/ssl_{pth}_gtv.npy"
        if switch or not os.path.exists(data_path):
            dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl/"
            extname = "/ckpt_v/model_best.pth"
            full_path = dirname + pth + extname
            res, length = get_res(full_path, learner)
            print(res)
            np.save(f"./tmp/tester_gtv/ssl_{pth}_gtv.npy", res)
            np.save(f"./tmp/tester_gtv/ssl_{pth}_count.npy", length)
        else:
            res = np.load(data_path)
            print(res)
        # import ipdb; ipdb.set_trace()
        plt.plot(np.arange(0, 7), res, label=pth)

    # ssl_probmap6
    pths = []
    pths += ["temp1.0"] # , "temp2.0"
    for pth in pths:
        data_path = f"./tmp/tester_gtv/ssl_probmap6_{pth}_gtv.npy"
        if switch or not os.path.exists(data_path):
            dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap6/"
            extname = "/ckpt_v/model_best.pth"
            full_path = dirname+pth+extname
            res, length = get_res(full_path, learner)
            print(res)
            np.save(f"./tmp/tester_gtv/ssl_probmap6_{pth}_gtv.npy", res)
            np.save(f"./tmp/tester_gtv/ssl_probmap6_{pth}_count.npy", length)
        else:
            res = np.load(data_path)
            print(res)
        # import ipdb; ipdb.set_trace()
        plt.plot(np.arange(0,7), res, label=pth)

    # plt.plot(np.arange(0,7), np.array(length) / np.array(length).mean(), label="thres")
    plt.legend()
    plt.savefig("./tmp/tmp.png")
    plt.savefig("./tmp/tmp.pdf")
