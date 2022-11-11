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

cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, nearby=None):
    b, w, h = cos_map.shape
    total_loss = list()
    gt_values_to_record = []
    for id in range(b):
        cos_map[id] = cos_map[id].exp()
        gt_value = cos_map[id, gt_x[id], gt_y[id]].clone()
        if nearby is not None:
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            chosen_patch = cos_map[id, min_x:max_x, min_y:max_y]
        else:
            chosen_patch = cos_map[id]
        id_loss = - torch.log(gt_value / chosen_patch.sum())
        gt_values_to_record.append(gt_value.clone().detach().log().cpu())
        total_loss.append(id_loss)
    gt_values_to_record = torch.stack(gt_values_to_record).mean()
    return torch.stack(total_loss).mean(), gt_values_to_record


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss, gt_values = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss, gt_values


def compare_vectors(ii, fea1, fea2, loc):
    scale = 2 ** (5-ii)
    raw_loc = loc // scale
    v1 = torch.stack([fea1[ii][[id], :, raw_loc[id][0], raw_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    v2 = torch.stack([fea2[ii][[id], :, raw_loc[id][0], raw_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    cos_sim = cosfn(v1.unsqueeze(-1).unsqueeze(-1), v2.unsqueeze(-1).unsqueeze(-1))
    cos_sim = torch.clamp(cos_sim, 0., 1.).mean()
    return cos_sim.squeeze()


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
        self.net2 = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
        # self.
        # self.net_patch = UNet_Pretrained(3, emb_len=16)
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()

    def forward(self, x, **kwargs):
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        crop_imgs = data['crop_imgs']
        raw_loc  = data['raw_loc']
        chosen_loc = data['chosen_loc']

        raw_fea_list1 = self.net(raw_imgs)
        raw_fea_list2 = self.net2(raw_imgs) # net_patch

        nearby = self.config['special']['nearby']
        loss_0 = compare_vectors(0, raw_fea_list1, raw_fea_list2, raw_loc)
        loss_1 = compare_vectors(1, raw_fea_list1, raw_fea_list2, raw_loc)
        loss_2 = compare_vectors(2, raw_fea_list1, raw_fea_list2, raw_loc)
        loss_3 = compare_vectors(3, raw_fea_list1, raw_fea_list2, raw_loc)
        loss_4 = compare_vectors(4, raw_fea_list1, raw_fea_list2, raw_loc)

        loss = loss_0
        res_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4}

        return res_dict

    def load(self, pth=None, pth2=None , *args, **kwargs):
        state_dict = torch.load(pth)
        self.net.load_state_dict(state_dict)
        state_dict = torch.load(pth2)
        self.net2.load_state_dict(state_dict)

    def save_optim(self, pth, optimizer, epoch, *args, **kwargs):
        pass

    def configure_optimizers(self, *args, **kwargs):
        config_train = self.config['training']
        optimizer = optim.Adam(params=self.net.parameters(), lr=config_train['lr'], betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=config_train['weight_decay'])
        # optimizer_patch = optim.Adam(params=self.net_patch.parameters(), lr=config_train['lr'],
        #                              betas=(0.9, 0.999), eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])
        # scheduler_patch = optim.lr_scheduler.StepLR(optimizer_patch, config_train['decay_step'], gamma=config_train['decay_gamma'])
        # return {'optimizer': [optimizer, optimizer_patch], 'scheduler': [scheduler, scheduler_patch]}
        return {'optimizer': [optimizer], 'scheduler': [scheduler]}


def test(logger, config):
    all_paths = []
    # all_paths.append(['prob5-400','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'])
    # all tested with best model
    # fake 224/ real 256
    all_paths.append(['ps-256', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4/ckpt_v/model_best.pth'])
    all_paths.append(['ps-196', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_2/ckpt_v/model_best.pth'])
    all_paths.append(['ps-128', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_3/ckpt_v/model_best.pth'])
    all_paths.append(['ps-96', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_4/ckpt_v/model_best.pth'])

    csvlogger = CSVLogger(config['base']['runs_dir'], mode="w")

    for p in all_paths:
        state_dict = _debug2(logger, config, p[0], p[1])
        state_dict['tag'] = p[0]
        csvlogger.record(state_dict)
        print_dict(state_dict)


def _debug2(logger, config, name, pth):
    upsample = 'nearest' # "nearest" / bilinear
    # tester_train = Tester(logger, config, split="Train", upsample=upsample, collect_sim=True)
    tester_test = Tester(logger, config, split="Test1+2", upsample=upsample,
                         collect_sim=True, collect_near=True)

    learner = Learner(logger=logger, config=config)
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
    res1['upsample'] = upsample
    return res1


def test_tp(logger, config):
    # test template
    all_paths = []

    # all tested with best model
    # fake 224/ real 256
    all_paths.append(['ps-256', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4/ckpt_v/model_best.pth'])
    all_paths.append(['ps-196', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_2/ckpt_v/model_best.pth'])
    all_paths.append(['ps-128', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_3/ckpt_v/model_best.pth'])
    all_paths.append(['ps-96', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_4/ckpt_v/model_best.pth'])

    upsample = 'bilinear' # "nearest" ， bilinear
    tester_train = Tester(logger, config, split="oneshot_debug", upsample=upsample,
                          collect_sim=True, collect_near=False)
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
        ids = [114, ]  # 114, 125 124,
        # for id_oneshot in ids:
        res1 = tester_train.test_template_sim(learner, oneshot_id=ids[0])
        # res1 = tester_test.test(learner, oneshot_id=ids[0])
        print_dict(res1)
        csvlogger.record(res1)


def diff_of_2_models(logger, config):
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    all_paths = []
    # all_paths.append(['ps-256', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4/ckpt_v/model_best.pth'])
    # all_paths.append(['ps-32', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps32/ckpt_v/model_best.pth'])
    all_paths.append(['ps-64', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps64/ckpt_v/model_best.pth'])
    all_paths.append(['ps-96', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_4/ckpt_v/model_best.pth'])
    # all_paths.append(['ps-128', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_3/ckpt_v/model_best.pth'])
    # all_paths.append(['ps-192', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_2/ckpt_v/model_best.pth'])

    learner = Learner(logger=logger, config=config)
    learner.load(all_paths[0][1], all_paths[1][1])
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id, collect_sim=False)
    # tester = None
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list = testset.ref_landmarks(template_oneshot_id)
    monitor = Monitor(key='mre', mode='dec')
    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=64,
                                  pre_crop=False, ref_landmark=landmark_list, use_prob=True, retfunc=2)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


def all_models(logger, config):
    all_paths = []

    # all tested with best model
    # fake 224/ real 256
    # all_paths.append(['ps-256', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4/'])
    # for pth_info in all_paths:
    #     csvlogger = CSVLogger(config['base']['runs_dir'], f"record_{pth_info[0]}.csv")
    #     tblogger = TBLogger(tdir(config['base']['runs_dir'], f"tb/{pth_info[0]}"))
    #     print(f"Using TBLogger! located in {tdir(config['base']['runs_dir'], f'tb/{pth_info[0]}')}")
    #     for epoch in range(50, 1000, 10):
    #         tag = pth_info[0] + "-" + str(epoch)
    #         path = pth_info[1] + f"ckpt/best_model_epoch_{epoch}.pth"
    #         if not os.path.exists(path):
    #             continue
    #         res = _debug2(logger, config, tag, path)
    #         res = {"tag": tag, **res}
    #         csvlogger.record(res)
    #         tblogger.record(res)
    # exit(0)

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
            # tblogger.record(res)


def draw_plot():
    pss = [32, 64, 96, 128, 160, 196, 224, 256, 288]
    # path_prefix = f"/home1/quanquan/code/landmark/code/runs/ana/ana_ps/all_models/record_ps-"
    path_prefix = "D:\Documents\git-clone\oneshot-landmark\landmark\code/runs/ana/ana_ps/all_models/record_ps-"
    data_dict = {f"ps-{ps}": pd.read_csv(path_prefix + str(ps) + ".csv") for ps in pss}

    # print(data1.dtypes)
    for name, data in data_dict.items():

        # for data, name in zip([data1, data2, data3], ['ip', 'mi', 'prob']):
        labels = []
        # labels += ['max_sim', 'lm_sim', 'sim_gap']
        # labels += ['sim_gap']
        labels += ['lm_sim']
        data['sim_gap'] = data['max_sim'] - data['lm_sim']
        data['id'] = range(len(data))

        # data['epoch'] = range(50, 50*len(data), len(data))
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


def ana_mi(logger, config):
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    all_paths = []
    # all tested with best model
    # fake 224/ real 256
    all_paths.append(['ps-256', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4/ckpt_v/model_best.pth'])
    all_paths.append(['ps-32', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps32/ckpt_v/model_best.pth'])
    all_paths.append(['ps-64', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_ps64/ckpt_v/model_best.pth'])
    all_paths.append(['ps-96', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_4/ckpt_v/model_best.pth'])
    all_paths.append(['ps-128', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_3/ckpt_v/model_best.pth'])
    all_paths.append(['ps-192', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/collect_sim_debug4_2/ckpt_v/model_best.pth'])

    upsample = 'nearest' # "nearest" ， bilinear
    # tester_train = Tester(logger, config, split="oneshot_debug", upsample=upsample,
    #                       collect_sim=True, collect_near=False)
    # tester_test = Tester(logger, config, split="Test1+2", upsample=upsample, collect_sim=True)
    # csvlogger = CSVLogger(config['base']['runs_dir'], mode="w")
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id, collect_sim=False)
    # tester = None
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list = testset.ref_landmarks(template_oneshot_id)

    ps_for_ana = 256

    for pth_info in all_paths:
        pth, name = pth_info[1], pth_info[0]
        learner = Learner(logger=logger, config=config)
        learner.load(pth)
        learner.cuda()
        learner.eval()
        monitor = Monitor(key='mre', mode='dec')
        dataset_train = Cephalometric(config['dataset']['pth'], patch_size=ps_for_ana,
                                      pre_crop=False, ref_landmark=landmark_list, use_prob=True, retfunc=2)
        trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
        learner = Learner(logger=logger, config=config)
        trainer.fit(learner, dataset_train)


if __name__ == '__main__':
    # draw_plot()
    # exit(0)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ana/ana.yaml")
    parser.add_argument('--tag', default="debug")
    parser.add_argument('--func', default="draw_plot")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print(config['base'])

    eval(args.func)(logger, config)
