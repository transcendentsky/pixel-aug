"""
    Analysis all metrics / params / indices , with different multi encoders

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
from tutils import TBLogger, print_dict


import seaborn as sns
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
# plt.style.use('ggplot')


torch.backends.cudnn.benchmark = True



def draw_plot(logger, config):
    import pandas as pd
    res = pd.read_csv("/home1/quanquan/code/landmark/code/runs/ana/ana_layer_ensemble/brute_search_in_4/record.csv")

    data_dict = {f"brute_search": res}
    for name, data in data_dict.items():
        labels = []
        # labels += ['max_sim', 'lm_sim', 'sim_gap']
        # labels += ['sim_gap']
        labels += ['lm_sim']
        data['sim_gap'] = data['max_sim'] - data['lm_sim']
        data['id'] = range(len(data))

        label_refered = "mre" # 'id', 'mre', 'epoch'
        for label in labels:
            # fig = sns.lineplot(x=label, y='mre', data=data, label=name + "_" + label)
            fig = sns.lineplot(x=label_refered, y=label, data=data, label=name + "_" + label)

    scatter_fig = fig.get_figure()
    ylabel = "" if len(labels) > 1 else "_" + labels[0]
    scatter_fig.savefig(f"ana_layer_{label_refered}{ylabel}"+".png", dpi=400)
    print("Saving")


def test(logger, config):
    all_paths = []
    # all tested with best model
    all_paths.append(['baseline-ps-64', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps64/'])
    all_paths.append(['224-64-64-64-64', '/home1/quanquan/code/landmark/code/runs/ssl2/multi_encoder/debug/'])  #
    all_paths.append(['64-64-32-32-32', '/home1/quanquan/code/landmark/code/runs/ssl2/multi_encoder/run4/'])  #
    all_paths.append(['64-64-64-32-32', '/home1/quanquan/code/landmark/code/runs/ssl2/multi_encoder/run3/'])  #
    all_paths.append(['64-64-64-64-32', '/home1/quanquan/code/landmark/code/runs/ssl2/multi_encoder/run5/'])  #
    all_paths.append(['96-64-64-32-64', '/home1/quanquan/code/landmark/code/runs/ssl2/multi_encoder/run6/'])  #

    csvlogger1 = CSVLogger(config['base']['runs_dir'], name="train.csv", mode="w")
    csvlogger2 = CSVLogger(config['base']['runs_dir'], name="test.csv", mode="w")
    split = ""
    for path in all_paths:
        name = path[0]
        ckpt_path = path[1] + f"ckpt_v/model_best.pth"
        dtrain, dtest = _test_a_model(ckpt_path, name, split)
        csvlogger1.record(dtrain)
        csvlogger2.record(dtest)
        print(dtest)


def _test_a_model(pth, name, split):
    tester_train = Tester(logger=logger, config=config, split="Train", default_oneshot_id=114,
                    upsample="bilinear",
                    collect_sim=True, collect_near=True)
    tester_test = Tester(logger=logger, config=config, split="Test1+2", default_oneshot_id=114,
                    upsample="bilinear",
                    collect_sim=True, collect_near=True)
    state_dict = torch.load(pth)
    net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
    net.load_state_dict(state_dict)
    net.cuda()
    res1 = tester_train.test(net=net, oneshot_id=114)
    res2 = tester_test.test(net=net, oneshot_id=114)
    return {"tag": name, **res1}, {"tag": name, **res2}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ana/ana.yaml")
    parser.add_argument('--tag', default="debug")
    parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print(config['base'])

    eval(args.func)(logger, config)
