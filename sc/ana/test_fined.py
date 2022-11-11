"""
    test with tester_fined.py /
        by upscaled global features

    Analysis of Max similarities between landmarks
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
import numpy as np
from tutils import CSVLogger, print_dict

# from utils.tester.tester_fined import Tester
# from datasets.ceph.ceph_ssl import Cephalometric
# from models.network_emb_study import UNet_Pretrained

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
        # self.net(x['img'])
        # raise NotImplementedError
        return self.net(x)

    def load(self, pth=None, force_match=False, *args, **kwargs):
        if pth is None:
            print("Load Pretrain Model")
            state_dict = torch.load(self.config['network']['pretrain'])
            self.net.load_state_dict(state_dict)
        else:
            print("Load Pretrain Model:", pth)
            if force_match:
                state_dict = self.net.state_dict()
                state_dict2 = torch.load(pth)
                for k, v in state_dict.items():
                    state_dict[k] = state_dict2[k]
                self.net.load_state_dict(state_dict)
            else:
                state_dict = torch.load(pth)
                self.net.load_state_dict(state_dict)



def test(logger, config):
    all_paths = []
    all_paths.append(['prob3', "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth"])
    all_paths.append(['prob5-400','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'])
    all_paths.append(['prob5-300','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_300.pth'])
    all_paths.append(['prob5-100','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_100.pth'])
    all_paths.append(['ip_poolsize4',"/home1/quanquan/code/landmark/code/runs/ssl/ssl_mi/pool_size4/ckpt_v/model_best.pth"])
    all_paths.append(['prob5_latest','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/model_latest.pth'])
    all_paths.append(['ip_debug5',"/home1/quanquan/code/landmark/code/runs/ssl/interpolate/debug5/ckpt/best_model_epoch_110.pth"])
    all_paths.append(['emd0', '/home1/quanquan/code/landmark/code/runs/ssl/emd/debug/ckpt_v/model_best.pth'])
    all_paths.append(['ip_collect_sim', '/home1/quanquan/code/landmark/code/runs/ssl/interpolate/collect_sim/ckpt_v/model_best.pth'])
    all_paths.append(['mi_collect_sim', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_mi/collect_sim/ckpt_v/model_best.pth'])
    all_paths.append(['v2_qs', "/home1/qsyao/oneshot-landmark/runs/map_coloraug_net_level_2/model_epoch_499.pth"])

    csvlogger = CSVLogger(config['base']['runs_dir'], mode="w")

    for p in all_paths:
        state_dict = _debug2(logger, config, p)
        state_dict['tag'] = p[0]
        csvlogger.record(state_dict)
        # print_dict(state_dict)


def _debug2(logger, config, pth_info):
    upsample = 'bilinear' # "nearest"
    tester_train = Tester(logger, config, split="Train", upsample=upsample, collect_sim=True)
    tester_test = Tester(logger, config, split="Test1+2", upsample=upsample, collect_sim=True)

    learner = Learner(logger=logger, config=config)
    pth, name = pth_info[1], pth_info[0]
    print(name, pth)
    _force_match = True if name == "v2_qs" else False
    learner.load(pth, force_match=_force_match)
    learner.cuda()
    learner.eval()

    # ids = [1,2,3,4,5,6,7,8,9]
    ids = [114, ]  # 114, 125 124,
    # for id_oneshot in ids:
    # res1 = tester_train.test(learner, oneshot_id=ids[0])
    res1 = tester_test.test(learner, oneshot_id=ids[0])
    return res1


def test_tp(logger, config):
    # test template
    all_paths = []
    # all_paths.append(['prob3', "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth"])
    # all_paths.append(['prob5-400','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'])
    # all_paths.append(['ps-192', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4/ckpt_v/model_best.pth'])
    # all_paths.append(['ps-192', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4/ckpt_v/model_best.pth'])

    all_paths.append(['ip_poolsize4',"/home1/quanquan/code/landmark/code/runs/ssl/ssl_mi/pool_size4/ckpt_v/model_best.pth"])
    all_paths.append(['prob5_latest','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/model_latest.pth'])
    all_paths.append(['ip_debug5',"/home1/quanquan/code/landmark/code/runs/ssl/interpolate/debug5/ckpt/best_model_epoch_110.pth"])
    all_paths.append(['emd0', '/home1/quanquan/code/landmark/code/runs/ssl/emd/debug/ckpt_v/model_best.pth'])
    all_paths.append(['prob3', "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth"])
    all_paths.append(['prob5-400','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'])
    all_paths.append(['prob5-300','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_300.pth'])
    all_paths.append(['prob5-100','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_100.pth'])
    all_paths.append(['mi_collect_sim', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_mi/collect_sim/ckpt_v/model_best.pth'])
    all_paths.append(['v2_qs', "/home1/qsyao/oneshot-landmark/runs/map_coloraug_net_level_2/model_epoch_499.pth"])

    upsample = 'nearest' # "nearest" ， bilinear
    tester_train = Tester(logger, config, split="oneshot_debug", upsample=upsample, collect_sim=True)
    # tester_test = Tester(logger, config, split="Test1+2", upsample=upsample, collect_sim=True)
    learner = Learner(logger=logger, config=config)

    csvlogger = CSVLogger(config['base']['runs_dir'], mode="w")
    for pth_info in all_paths:
        pth, name = pth_info[1], pth_info[0]
        print(name, pth)
        _force_match = True if name == "v2_qs" else False
        learner.load(pth, force_match=_force_match)
        learner.cuda()
        learner.eval()

        # ids = [1,2,3,4,5,6,7,8,9]
        ids = [114, ]  # 114, 125 124,
        # for id_oneshot in ids:
        res1 = tester_train.test_template_sim(learner, oneshot_id=ids[0])
        # res1 = tester_test.test(learner, oneshot_id=ids[0])
        print_dict(res1)
        res1 = {"tag": name, **res1}
        csvlogger.record(res1)
        # import ipdb; ipdb.set_trace()
        # return res1

def draw_plot(logger=None, config=None):
    import pandas as pd
    # compare with regular results
    data_dict = {}
    data_dict['data_fined'] = pd.read_csv("D:\Documents\git-clone\oneshot-landmark\landmark\code/runs/ana/test_fined/test12/record.csv")
    data_dict['data_linear'] = pd.read_csv("D:\Documents\git-clone\oneshot-landmark\landmark\code/runs/ana/max_sim/collect_sim/record_test_linear.csv")
    data_dict['data_nearest'] = pd.read_csv("D:\Documents\git-clone\oneshot-landmark\landmark\code/runs/ana/max_sim/collect_sim/record_test_near.csv")

    for name, data in data_dict.items():
        # for data, name in zip([data1, data2, data3], ['ip', 'mi', 'prob']):
        labels = []
        # labels += ['max_sim', 'lm_sim', 'sim_gap']
        # labels = ['sim_gap']
        # labels = ['lm_sim']
        data['sim_gap'] = data['max_sim'] - data['lm_sim']
        data['id'] = range(len(data))
        # data['epoch'] = range(50, 1000, 50)
        # for i in range(5):
        #     data[f'sim_l_gap_{i}'] = data[f'sim_layer_max_{i}'] - data[f'sim_layer_point_{i}']
        # data[f'sim_l_gap_sum'] = data[f'sim_l_gap_0'] + data[f'sim_l_gap_1'] + data[f'sim_l_gap_2'] + data[f'sim_l_gap_3'] + data[f'sim_l_gap_4']
        # data[f'sim_layer_point_sum'] = data[f'sim_layer_point_{0}'] + data[f'sim_layer_point_{1}'] + data[
        #     f'sim_layer_point_{2}'] + data[f'sim_layer_point_{3}'] + data[f'sim_layer_point_{4}']
        # data[f'sim_layer_max_sum'] = data[f'sim_layer_max_{0}'] + data[f'sim_layer_max_{1}'] + data[
        #     f'sim_layer_max_{2}'] + data[f'sim_layer_max_{3}'] + data[f'sim_layer_max_{4}']

        # labels += ['mean_sim']
        # labels += ['max_sim']
        # labels += [f'sim_l_gap_{i}' for i in range(5)]
        # labels += [f'sim_l_gap_sum']
        # labels += [f'sim_l_gap_0']
        labels += ['mre']
        # labels += [f'sim_layer_point_sum']
        # labels += [f'sim_layer_max_sum']
        label_refered = "id" # 'id', 'mre', 'epoch'
        for label in labels:
            # fig = sns.lineplot(x=label, y='mre', data=data, label=name + "_" + label)
            fig = sns.lineplot(x=label_refered, y=label, data=data, label=name + "_" + label)

    scatter_fig = fig.get_figure()
    ylabel = "" if len(labels) > 1 else "_" + labels[0]
    scatter_fig.savefig(f"test_fined_{label_refered}{ylabel}"+".png", dpi=400)



if __name__ == '__main__':
    draw_plot()
    exit(0)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ana/ana.yaml")
    parser.add_argument('--tag', default="debug")
    parser.add_argument('--func', default="test_tp")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print(config['base'])
    print("Using func: ", args.func)
    eval(args.func)(logger, config)
