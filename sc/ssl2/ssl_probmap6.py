"""
    only entropy map, with temperature
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl_entropy import Cephalometric
from models.network_emb_study import UNet_Pretrained
from einops import rearrange, repeat
import random
import numpy as np
import cv2


torch.backends.cudnn.benchmark = True

EX_CONFIG = {
    "dataset":{
        'entr': '/home1/quanquan/datasets/Cephalometric/entr1/train/',
        'prob': '/home1/quanquan/datasets/Cephalometric/prob_ks1/train/',
    },
    "special":{
        "patch_size": 64,
        "prob_ratio": 0,
        "cj_brightness": 0.15, # 0.8,  # 0.15
        "cj_contrast": 0.25, # 0.6,  # 0.25
        "entr_t": 0,
        'entr_weight': 0.7,
        'threshold': 4,
        'inverse': False,
        'num_repeat': 10,
        'thresholds': [0, 3],
        # 0, 2.6
        # 2, 3.15
        # 3, 3.708
        # 4, 4.57
        # 4.5, 5.75
    },
    "training": {
        'save_mode': ['best', 'latest'],
        'num_epochs': 600,
        'lr': 0.0001,
    }
}

def train(logger, config):
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    # ------------------------
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id, collect_sim=False, split="Test1+2", upsample="bilinear")
    monitor = Monitor(key='mre', mode='dec')

    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    all_landmarks = []
    for i in range(len(testset)):
        landmark_list = testset.ref_landmarks(i)
        all_landmarks.append(landmark_list)
    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['special']['patch_size'],
                                  cj_brightness=config['special']['cj_brightness'], cj_contrast=config['special']['cj_contrast'],
                                    entr_map_dir=config['dataset']['entr'], prob_map_dir=config['dataset']['prob'],
                                    mode="Train", retfunc=2, hard_select=True, num_repeat=config['special']['num_repeat'], runs_dir=config['base']['runs_dir'])
    # dataset_train.prob_map_for_all(all_landmarks)
    # ratio = dataset_train.entr_map_from_image4(thresholds=config['special']['thresholds'], inverse=True)
    ratio = dataset_train.entr_map_from_image3(thresholds=config['special']['thresholds'], inverse=True)
    logger.info(f"Used prob map areas percentage: {ratio}, thresholds: {config['special']['thresholds']}")
    # exit(0)
    # dataset_train.entr_map_from_image_old(temperature=config['special']['entr_t'], inverse=config['special']['inverse'])
    # ratio = dataset_train.entr_map_from_image2(threshold=config['special']['threshold'], temperature=config['special']['entr_t'], weight=config['special']['entr_weight'])
    # logger.info(f"high entr ratio: {ratio}")
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)

cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
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
        # entr = data['entr']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list = self.net_patch(crop_imgs)

        nearby = self.config['special']['nearby']
        loss_0, gt_value_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
        loss_1, gt_value_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_2, gt_value_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_3, gt_value_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_4, gt_value_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)

        # import ipdb; ipdb.set_trace()
        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
        gtv = gt_value_0 * gt_value_1 * gt_value_2 * gt_value_3 * gt_value_4
        # gtv = gtv / 5
        # ratio = gtv / entr

        loss_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3,
                     'loss_4': loss_4}
        gtvs_dict = {'gtv': gtv, 'gtv_0': gt_value_0, 'gtv_1': gt_value_1, 'gtv_2': gt_value_2, 'gtv_3': gt_value_3,
                     'gtv_4': gt_value_4}
        # ratio_dict = {'gtv_vs_entr': ratio}
        res_dict = {**loss_dict, **gtvs_dict}

        return res_dict

    def load(self, pth=None, *args, **kwargs):
        if pth is None:
            print("Load Pretrain Model")
            state_dict = torch.load(self.config['network']['pretrain'])
            self.net.load_state_dict(state_dict)
        else:
            print("Load Pretrain Model:", pth)
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




def test(logger, config):
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id, collect_sim=False, split="Test1+2", upsample="bilinear")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print_dict(config)
    # import ipdb; ipdb.set_trace()

    eval(args.func)(logger, config)
