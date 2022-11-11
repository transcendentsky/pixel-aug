"""
    Analysis of Max similarities between landmarks
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl import Cephalometric
from models.network_emb_study import UNet_Pretrained
import numpy as np
from tutils import CSVLogger, print_dict


torch.backends.cudnn.benchmark = True


cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, nearby=None, get_mi=False):
    b, w, h = cos_map.shape
    total_loss = list()
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
        if get_mi:
            log_K = 1.20411 # log(16)
            # import ipdb; ipdb.set_trace()
            i_nce = log_K * b - id_loss
            id_loss = i_nce
        total_loss.append(id_loss)
    return torch.stack(total_loss).mean()


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None, get_mi=False):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby, get_mi=get_mi)
    return loss


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

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        crop_imgs = data['crop_imgs']
        raw_loc  = data['raw_loc']
        chosen_loc = data['chosen_loc']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list = self.net_patch(crop_imgs)

        get_mi = True
        nearby = self.config['special']['nearby']
        loss_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, get_mi=get_mi)
        loss_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, get_mi=get_mi)
        loss_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, get_mi=get_mi)
        loss_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, get_mi=get_mi)
        loss_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, get_mi=get_mi)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

        return {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4}

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


    def save_optim(self, pth, optimizer, epoch, *args, **kwargs):
        pass

    def configure_optimizers(self, *args, **kwargs):
        config_train = self.config['training']
        optimizer = optim.Adam(params=self.net.parameters(), lr=config_train['lr'], betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=config_train['weight_decay'])
        optimizer_patch = optim.Adam(params=self.net_patch.parameters(), lr=config_train['lr'],
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])
        scheduler_patch = optim.lr_scheduler.StepLR(optimizer_patch, config_train['decay_step'], gamma=config_train['decay_gamma'])
        return {'optimizer': [optimizer, optimizer_patch], 'scheduler': [scheduler, scheduler_patch]}


# def train(logger, config):
#     from datasets.ceph.ceph_ssl import Test_Cephalometric
#     tester = Tester(logger, config)
#     monitor = Monitor(key='mre', mode='dec')
#
#     id_oneshot = 114
#     testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
#     landmark_list = testset.ref_landmarks(id_oneshot)
#     dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['training']['patch_size'],
#                                   pre_crop=False, ref_landmark=landmark_list, use_prob=True, retfunc=2)
#     trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
#     learner = Learner(logger=logger, config=config)
#     trainer.fit(learner, dataset_train)


def ince(logger, config):
    """ Calc I_nce (mutual info), refer to <What makes for good views for CL> """
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    config['training']['num_epochs'] = 2
    config['training']['start_epoch'] = 1
    tester = Tester(logger, config)
    monitor = Monitor(key='mre', mode='dec')
    id_oneshot = 114
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list = testset.ref_landmarks(id_oneshot)
    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['training']['patch_size'],
                                  pre_crop=False, ref_landmark=landmark_list, use_prob=True, retfunc=2)
    trainer = Trainer(logger=logger, config=config, tester=None, monitor=monitor)
    learner = Learner(logger=logger, config=config)

    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl/test_qsyao/ckpt/best_model_epoch_100.pth')

    learner.load("/home1/qsyao/oneshot-landmark/runs/map_coloraug_net_level_2/model_epoch_499.pth", force_match=True)
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl/test_qsyao/ckpt/best_model_epoch_100.pth')
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl/train1/ckpt/best_model_epoch_390.pth')
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl_ensemble/train2/ckpt/best_model_epoch_540.pth')
    # learner.load("/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth")
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth')

    trainer.fit(learner, dataset_train)



def resolve_cos_list(cos_list_list_list):
    temp = np.array(cos_list_list_list)
    # shape (150, 19, 5, 384* 384)
    final_cos = temp[:, :, -1, :]
    pass


def resolve_max_sim_list(max_list):
    temp = np.array(max_list)
    mean = np.mean(temp)
    # import ipdb; ipdb.set_trace()
    return mean


def resolve_max_sim_layer(max_list):
    # max_list: 150, 19, 5
    temp = np.array(max_list)
    # import ipdb; ipdb.set_trace()
    mean = np.mean(temp, axis=0)
    mean = np.mean(mean, axis=0)
    return mean


def test(logger, config):
    all_paths = []
    # all_paths.append(['prob3', "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth"])
    all_paths.append(['prob5-400','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'])
    # all_paths.append(['prob5-300','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_300.pth'])
    # all_paths.append(['prob5-100','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_100.pth'])
    # all_paths.append(['v2_qs', "/home1/qsyao/oneshot-landmark/runs/map_coloraug_net_level_2/model_epoch_499.pth"])
    # all_paths.append(['ip_debug_bad',"/home1/quanquan/code/landmark/code/runs/ssl/interpolate/debug/ckpt_v/model_best.pth"])
    # all_paths.append(['ip_poolsize4',"/home1/quanquan/code/landmark/code/runs/ssl/ssl_mi/pool_size4/ckpt_v/model_best.pth"])
    # all_paths.append(['prob5_latest','/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/model_latest.pth'])
    # all_paths.append(['ip_debug4_bad',"/home1/quanquan/code/landmark/code/runs/ssl/interpolate/debug4/ckpt_v/model_best.pth"])
    # all_paths.append(['ip_debug5',"/home1/quanquan/code/landmark/code/runs/ssl/interpolate/debug5/ckpt/best_model_epoch_110.pth"])
    # all_paths.append(['emd0', '/home1/quanquan/code/landmark/code/runs/ssl/emd/debug/ckpt_v/model_best.pth'])
    # all_paths.append(['ip_collect_sim', '/home1/quanquan/code/landmark/code/runs/ssl/interpolate/collect_sim/ckpt_v/model_best.pth'])
    # all_paths.append(['mi_collect_sim', '/home1/quanquan/code/landmark/code/runs/ssl/ssl_mi/collect_sim/ckpt_v/model_best.pth'])

    csvlogger = CSVLogger(config['base']['runs_dir'], mode="w")

    for p in all_paths:
        state_dict = _debug2(logger, config, p)
        state_dict['tag'] = p[0]
        csvlogger.record(state_dict)
        print_dict(state_dict)


def _debug2(logger, config, pth_info):
    upsample = 'nearest' # "nearest"
    # tester_train = Tester(logger, config, split="Train", upsample=upsample, collect_sim=True)
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
    res1['upsample'] = upsample
    return res1


def _debug(logger, config, pth_info):
    upsample = "nearest"
    tester_train = Tester(logger, config, split="Train", upsample=upsample, collect_sim=True)
    tester_test  = Tester(logger, config, split="Test1+2", upsample=upsample, collect_sim=True)

    learner = Learner(logger=logger, config=config)
    # learner.load("/home1/qsyao/oneshot-landmark/runs/map_coloraug_net_level_2/model_epoch_499.pth", force_match=True)
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl/test_qsyao/ckpt/best_model_epoch_100.pth')
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl/train1/ckpt/best_model_epoch_390.pth')
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl_ensemble/train2/ckpt/best_model_epoch_540.pth')
    # learner.load("/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth")
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth')
    # learner.load(tfilename(config['base']['runs_dir'], 'ckpt', 'best_model_epoch_400.pth'))
    # learner.load("/home1/quanquan/code/landmark/code/runs/ana/finetune3/pretrain_prob3/ckpt_v/model_best.pth", force_match=True)

    pth, name = pth_info
    # learner.load(pth, force_match=True)
    learner.load(pth)
    learner.cuda()
    learner.eval()

    # ids = [1,2,3,4,5,6,7,8,9]
    ids = [114, ] # 114, 125 124,
    for id_oneshot in ids:
        # logger.info("Test1")
        # tester_test1 = Tester(logger, config, split="Test1+2")
        # res0 = tester_test1.test(learner, oneshot_id=id_oneshot)
        # logger.info(res0)
        # import ipdb; ipdb.set_trace()

        if True:
            res1 = tester_train.test(learner, oneshot_id=id_oneshot, collect_details=True)
            mean_max1 = resolve_max_sim_list(res1['max_sim'])
            mean_lm1 = resolve_max_sim_list(res1['lm_sim'])
            mean_max2_1 = resolve_max_sim_layer(res1['max_sim_layer1'])
            mean_max3_1 = resolve_max_sim_layer(res1['max_sim_layer2'])
            logger.info("Train")
            logger.info(f"mean_max_sim: {mean_max1}, mean_lm_sim:{mean_lm1}, mre: {res1['mre']}")
            logger.info(f"sim_layer1: {mean_max2_1}, sim_layer2: {mean_max3_1}")
            # import ipdb; ipdb.set_trace()
            # logger.info(res1)

            res2 = tester_test.test(learner, oneshot_id=id_oneshot, collect_details=True)
            mean_max2 = resolve_max_sim_list(res2['max_sim'])
            mean_lm2 = resolve_max_sim_list(res2['lm_sim'])
            mean_max2_2 = resolve_max_sim_layer(res2['max_sim_layer1'])
            mean_max3_2 = resolve_max_sim_layer(res2['max_sim_layer2'])
            logger.info("Test")
            logger.info(f"mean_max_sim: {mean_max2}, mean_lm_sim: {mean_lm2}, mre: {res2['mre']}")
            logger.info(f"sim_layer1: {mean_max2_2}, sim_layer2: {mean_max3_2}")
            # logger.info(res2)
            # import ipdb; ipdb.set_trace()

        state_dict = {"Train-max-sim": mean_max1, "Train-lm-sim": mean_lm1, "Train-mre": res1['mre'],
                      "Test12-max-sim": mean_max2, "Test12-lm-sim": mean_lm2, "Test12-mre": res2['mre'],
                      "pth": pth, "id_oneshot": id_oneshot}


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
        res1 = tester_train.test_template_sim(learner, oneshot_id=ids[0], draw=True)
        # res1 = tester_test.test(learner, oneshot_id=ids[0])
        res1 = {"tag": name, **res1}
        # print_dict(res1)
        csvlogger.record(res1)
        # import ipdb; ipdb.set_trace()
        # return res1


if __name__ == '__main__':
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
