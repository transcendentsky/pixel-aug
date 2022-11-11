"""
    Analysis all metrics / params / indices , with different patch sizes

    Analysis of Max similarities between landmarks
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
from utils.tester.tester_ssl_layer_ensemble import Tester
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


def test(logger, config):
    csvlogger = CSVLogger(config['base']['runs_dir'], mode="w")

    # all_gates = [str(i*32) for i in range(2, 10)]
    # for i in range(9):
    #     gates = ["64", "64", "64", "64", all_gates[i]]
    #     res = _test_once(logger, config, gates, split)
    #     # print_dict(res)
    #     csvlogger.record(res)
    #     print(gates, res['mre'])
    #
    # for i in range(9):
    #     gates = [all_gates[i], "64", "64", "64", "64"]
    #     res = _test_once(logger, config, gates, split)
    #     # print_dict(res)
    #     csvlogger.record(res)
    #     print(gates, res['mre'])
    #
    # for i in range(9):
    #     gates = ["64", all_gates[i], "64", "64", "64"]
    #     res = _test_once(logger, config, gates, split)
    #     # print_dict(res)
    #     csvlogger.record(res)
    #     print(gates, res['mre'])
    #
    # for i in range(9):
    #     gates = ["64", "64", all_gates[i], "64", "64"]
    #     res = _test_once(logger, config, gates, split)
    #     # print_dict(res)
    #     csvlogger.record(res)
    #     print(gates, res['mre'])
    #
    # for i in range(9):
    #     gates = ["64", "64", "64", all_gates[i] ,"64"]
    #     res = _test_once(logger, config, gates, split)
    #     # print_dict(res)
    #     csvlogger.record(res)
    #     print(gates, res['mre'])

    # -------------
    split = "Train"
    # tag_list = ["64", "96", "160", "224"]
    tag_list = ["32", "64"]
    for l1 in tag_list:
        for l2 in tag_list:
            for l3 in tag_list:
                for l4 in tag_list:
                    for l5 in tag_list:
                        gates = [l1, l2, l3, l4, l5]
                        res = _test_once(logger, config, gates, split)
                        csvlogger.record(res)
                        print(gates, res['mre'])



def _test_once(logger, config, gates, split):
    all_paths = []
    # all tested with best model
    all_paths.append(['ps-32', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps32/'])  #
    all_paths.append(['ps-64', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps64/'])
    all_paths.append(['ps-96', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_4/'])
    all_paths.append(['ps-128', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_3/'])
    all_paths.append(['ps-160', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_8/'])
    all_paths.append(['ps-192', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_2/'])
    all_paths.append(['ps-224', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_5/'])
    all_paths.append(['ps-256', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_6/'])
    all_paths.append(['ps-288', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_7/'])

    new_all_paths = []
    for g in set(gates):
        for path in all_paths:
            name = path[0][3:]
            if name == g:
                new_all_paths.append(path)
    all_paths = new_all_paths
    print("For saving memory, only load: ", all_paths)

    tester = Tester(logger=logger, config=config, split=split, default_oneshot_id=114,
                    upsample="bilinear",
                    collect_sim=True, collect_near=True)
    nets = {}
    for path in all_paths:
        name = path[0][3:]
        ckpt_path = path[1] + f"ckpt_v/model_best.pth"
        state_dict = torch.load(ckpt_path)
        net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
        net.load_state_dict(state_dict)
        nets[f"{name}"] = net.cuda()
        # nets[f'{name}'] = "for debug"

    res = tester.test(nets=nets, oneshot_id=114, gates=gates)
    res = {"gates": gates, **res}
    return res


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
