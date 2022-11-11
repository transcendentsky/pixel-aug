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
    tester = Tester(logger, config, retfunc=2)
    learner.load(pth)
    learner.cuda()
    learner.eval()
    res = tester.test(learner)
    return res


mi_self = [2.51138927, 3.18064627, 3.1936682, 2.11334156, 3.26182041, 2.72902131,
           2.16312098, 1.94328915, 2.01532239, 3.26112403, 3.20813622, 3.16552408,
           3.12697808, 2.64613941, 3.25124197, 1.97482131, 3.206975,  3.22484813,
           2.66378744]
mi_inter = [0.58855864, 0.9689842 , 0.88423108, 0.41595766, 0.93467499, 0.87520715,
            0.75355031, 0.69097756, 0.70706193, 0.9617875 , 0.95781572, 0.96094506,
            0.98639255, 0.89073955, 0.96499311, 0.71407015, 0.93329412, 0.91891464,
            0.7271314 ]

mre = [1.399, 1.5526,1.7562,2.9915 , 1.7145, 1.9027,
       1.8260, 1.3118, 1.1585, 4.4261, 2.2153, 2.8174,
       1.5348, 1.9849, 1.3075, 1.8713,1.5415, 3.7358,
       2.3073]
mre2 =[1.6767,1.7246,2.2157,2.5067,1.9819,2.2091,
       1.7173,1.3327,1.6392,2.2740,1.6451,1.7791,
       1.5442,1.7209,1.8208,2.9706,1.8948,1.9768,
       2.1830]


if __name__ == '__main__':
    fig = plt.figure()
    plt.plot(np.arange(0,19), mre, label="mre1")
    plt.plot(np.arange(0,19), mre2, label="mre2")
    # plt.bar(np.arange(0,5), np.arange(0,5))
    # plt.savefig("./tmp/tmp.png")
    # exit()
    switch = False
    pths = []
    # pths += ["edge_inv"]
    pths += ['gaussian_ps64_pseudo']
    # pths += ["entr_ps64_inv2", "entr_t0.3_(p6)"]
    # pths += ["prob_ks10", "entr_new_ushape"]
    learner = Learner(logger, config)
    length = np.array([1,1,1,1,1,1])
    for pth in pths:
        data_path = f"./tmp/tester_gtv2/ssl_probmap7_{pth}_gtv.npy"
        if switch or not os.path.exists(data_path):
            dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap7/"
            extname = "/ckpt_v/model_best.pth"
            full_path = dirname+pth+extname
            res, length = get_res(full_path, learner)
            print(res)
            np.save(f"./tmp/tester_gtv2/ssl_probmap7_{pth}_gtv.npy", res)
            np.save(f"./tmp/tester_gtv2/ssl_probmap7_{pth}_count.npy", length)
        else:
            res = np.load(data_path)
            print(res)
        # import ipdb; ipdb.set_trace()
        plt.plot(np.arange(0,19), res, label=pth)

    plt.plot(np.arange(0, 19), mi_self / np.array(mi_self).mean() * np.array(mi_inter).mean(), label='mi_self')
    plt.plot(np.arange(0, 19), mi_inter, label='mi_inter')
    plt.legend()
    plt.savefig("./tmp/tmp2.png")
    plt.savefig("./tmp/tmp2.pdf")
    # exit()
    # plt.legend()
    # plt.savefig("./tmp/tmp.png")
    exit()

    # ssl
    pths = []
    pths += ["baseline_ps64_3"]
    for pth in pths:
        data_path = f"./tmp/tester_gtv2/ssl_{pth}_gtv.npy"
        if switch or not os.path.exists(data_path):
            dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl/"
            extname = "/ckpt_v/model_best.pth"
            full_path = dirname + pth + extname
            res, length = get_res(full_path, learner)
            print(res)
            np.save(f"./tmp/tester_gtv2/ssl_{pth}_gtv.npy", res)
            np.save(f"./tmp/tester_gtv2/ssl_{pth}_count.npy", length)
        else:
            res = np.load(data_path)
            print(res)
        # import ipdb; ipdb.set_trace()

        plt.plot(np.arange(0, 19), res, label=pth)

    # ssl_probmap6
    pths = []
    pths += ["temp1.0", "temp2.0"]
    for pth in pths:
        data_path = f"./tmp/tester_gtv2/ssl_probmap6_{pth}_gtv.npy"
        if switch or not os.path.exists(data_path):
            dirname = "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap6/"
            extname = "/ckpt_v/model_best.pth"
            full_path = dirname+pth+extname
            res, length = get_res(full_path, learner)
            print(res)
            np.save(f"./tmp/tester_gtv2/ssl_probmap6_{pth}_gtv.npy", res)
            np.save(f"./tmp/tester_gtv2/ssl_probmap6_{pth}_count.npy", length)
        else:
            res = np.load(data_path)
            print(res)
        # import ipdb; ipdb.set_trace()

        plt.plot(np.arange(0,19), res, label=pth)

    # plt.plot(np.arange(0,7), np.array(length) / np.array(length).mean(), label="thres")
    plt.legend()
    plt.savefig("./tmp/tmp2.png")
    plt.savefig("./tmp/tmp2.pdf")
