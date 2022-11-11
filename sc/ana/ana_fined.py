"""
    Analysis of fined trained cos map

"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
import numpy as np
from tutils import CSVLogger, print_dict, tfilename, tdir
import pandas as pd
import os

# --------
from utils.tester.tester_ssl_debug import Tester
from utils.tester.tester_fined import Tester as Tester_fined
from datasets.ceph.ceph_ssl import Cephalometric
from models.network_emb_study import UNet_Pretrained
from tutils import TBLogger

import seaborn as sns
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
# plt.style.use('ggplot')


torch.backends.cudnn.benchmark = True


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
        self.net_patch = UNet_Pretrained(3, emb_len=16)
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()

    def forward(self, x, **kwargs):
        return self.net(x)

    def load(self, pth=None, force_match=False, *args, **kwargs):
        state_dict = torch.load(pth)
        self.net.load_state_dict(state_dict)


def test(logger, config):
    all_paths = []
    # fake 224/ real 256
    all_paths.append(['fined', '/home1/quanquan/code/landmark/code/runs/ssl2/fined2/run2/'])
    all_paths.append(['ps-64-baseline', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps64/'])

    csvlogger1 = CSVLogger(config['base']['runs_dir'], name="train.csv", mode="w")
    csvlogger2 = CSVLogger(config['base']['runs_dir'], name="test.csv", mode="w")
    for p in all_paths:
        pth = p[1] + f"ckpt_v/model_best.pth"
        dtrain, dtest = _debug2(logger, config, name=p[0], pth=pth)
        csvlogger1.record(dtrain)
        csvlogger2.record(dtest)
        print_dict(dtest)


def _debug2(logger, config, name, pth):
    upsample = 'bilinear' # "nearest" / bilinear
    tester_train = Tester(logger, config, split="Train", upsample=upsample, collect_sim=True)
    tester_test = Tester(logger, config, split="Test1+2", upsample=upsample,
                         collect_sim=True, collect_near=True)

    learner = Learner(logger=logger, config=config)
    learner.load(pth)
    learner.cuda()
    learner.eval()
    ids = [114, ]  # 114, 125 124,
    # for id_oneshot in ids:
    res1 = tester_train.test(learner, oneshot_id=ids[0])
    res2 = tester_test.test(learner, oneshot_id=ids[0])
    return {"tag": name, **res1}, {"tag": name, **res2}



def all_models(logger, config):
    all_paths = []
    # all_paths.append(['ps-196', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_2/'])
    # all_paths.append(['ps-128', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_3/'])
    # all_paths.append(['ps-96', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_4/'])
    # all_paths.append(['ps-224', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_5/'])
    all_paths.append(['ps-256', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_6/'])
    all_paths.append(['ps-288', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_7/'])
    all_paths.append(['ps-160', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_8/'])
    # all_paths.append(['ps-64', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps64/'])
    # all_paths.append(['ps-32', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps32/'])

    for pth_info in all_paths:
        csvlogger = CSVLogger(config['base']['runs_dir'], f"record_{pth_info[0]}.csv")
        tblogger = TBLogger(tdir(config['base']['runs_dir'], f"tb/{pth_info[0]}"))
        print(f"Using TBLogger! located in {tdir(config['base']['runs_dir'], f'tb/{pth_info[0]}')}")
        for epoch in range(50, 800, 50):
            tag = pth_info[0] + "-" + str(epoch)
            path = pth_info[1] + f"ckpt/model_epoch_{epoch}.pth"
            if not os.path.exists(path):
                print(f"Path not exists! {path}")
                break
            res = _debug2(logger, config, tag, path)
            res = {"tag": tag, **res}
            csvlogger.record(res)


def draw_plot():
    pss = [32, 64, 96, 128, 160, 196, 224, 256, 288]
    path_prefix = "D:\Documents\git-clone\oneshot-landmark\landmark\code/runs/ana/ana_ps/all_models/record_ps-"
    data_dict = {f"ps-{ps}": pd.read_csv(path_prefix + str(ps) + ".csv") for ps in pss}

    for name, data in data_dict.items():

        # for data, name in zip([data1, data2, data3], ['ip', 'mi', 'prob']):
        labels = []
        # labels += ['max_sim', 'lm_sim', 'sim_gap']
        # labels += ['sim_gap']
        labels += ['lm_sim']
        data['sim_gap'] = data['max_sim'] - data['lm_sim']
        data['id'] = range(len(data))

        # labels += ['mean_sim']
        # labels += ['max_sim']
        # labels += [f'sim_l_gap_{i}' for i in range(5)]
        # labels += [f'sim_l_gap_sum']
        # labels += [f'sim_l_gap_0']
        # labels += ['mre']
        # labels += [f'sim_layer_point_sum']
        # labels += [f'sim_layer_max_sum']
        label_refered = "id" # 'id', 'mre', 'epoch'
        for label in labels:
            # fig = sns.lineplot(x=label, y='mre', data=data, label=name + "_" + label)
            fig = sns.lineplot(x=label_refered, y=label, data=data, label=name + "_" + label)

    scatter_fig = fig.get_figure()
    ylabel = "" if len(labels) > 1 else "_" + labels[0]
    scatter_fig.savefig(f"ana_ps_{label_refered}{ylabel}"+".png", dpi=400)



if __name__ == '__main__':
    # draw_plot()
    # exit(0)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ana/ana.yaml")
    parser.add_argument('--tag', default="debug")
    parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print(config['base'])

    eval(args.func)(logger, config)
