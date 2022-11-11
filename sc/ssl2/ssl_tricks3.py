"""
    ssl_probmap3 + neg interpolate
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl_adap import Cephalometric
from models.network_emb_study import UNet_Pretrained
from einops import rearrange, repeat
import random
import numpy as np


torch.backends.cudnn.benchmark = True

EX_CONFIG = {
    "special": {
        "cj_brightness": 1.6, # 0.15
        "cj_contrast": 1.2, # 0.25
        "cj_saturation": 0.3, # 0.
        "cj_hue": 0.2 # 0
    }
}

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


def _tmp_get_neg_loc(gt_h, gt_w, shape, nearby=None):
    h, w = shape
    while True:
        if nearby is not None:
            d_h = random.choice([ 2,  -2, ])
            d_w = random.choice([ 2,  -2, ])
        else:
            d_h = random.choice([1, -1])
            d_w = random.choice([1, -1])
        if gt_h+d_h >=0 and gt_h + d_h < h:
            if gt_w + d_w >= 0 and gt_w + d_w < w:
                break
    return d_h, d_w


def extrapolate_loss(feature, tmpl_feature, gt_h, gt_w, nearby):
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)
    lam = lam if lam < 0.5 else 1-lam
    b, c, h, w = feature.shape
    ratio = 1.0 - lam
    ratio2 = 1.0 + lam

    b = tmpl_feature.shape[0]
    total_loss = []
    total_loss2 = []
    gt_values_to_record = []
    # import ipdb; ipdb.set_trace()
    # print(feature.shape)
    for bi in range(b):
        # print(feature.shape[-2:])
        d_h, d_w = _tmp_get_neg_loc(gt_h[bi], gt_w[bi], shape=feature.shape[-2:], nearby=nearby)

        vector_tmp = tmpl_feature[bi]
        # vector_feature = feature[bi, :, gt_h[bi], gt_w[bi]]
        neg_vector = feature[bi, :, gt_h[bi]+d_h, gt_w[bi]+d_w]
        if nearby is not None:
            min_x, max_x = max(gt_h[bi] - nearby, 0), min(gt_h[bi] + nearby, w)
            min_y, max_y = max(gt_w[bi] - nearby, 0), min(gt_w[bi] + nearby, h)
        else:
            min_x, max_x = 0, h
            min_y, max_y = 0, w

        afeature_map = feature[bi]
        afeature_map = afeature_map * ratio + repeat(neg_vector, "c -> c h w", h=h, w=w) * (1-ratio)
        # vector_tmp2 = vector_tmp * ratio2 + vector_feature * (1-ratio2)
        # vector_feature2 = vector_feature * ratio2 + vector_tmp * (1-ratio2)

        # cos_map  = match_inner_product(afeature_map.unsqueeze(0), vector_tmp2.unsqueeze(0))
        # vector_sim = match_inner_product(rearrange(vector_feature2, "c -> 1 c 1 1"), rearrange(vector_tmp2, "c -> 1 c"))
        # iloss = - torch.log(vector_sim.exp() / cos_map.exp().sum())

        ### interpolate
        # cos_map2 = match_inner_product(afeature_map2.unsqueeze(0), vector_tmp.unsqueeze(0))
        # vector_sim2 = match_inner_product(rearrange(vector_feature, "c -> 1 c 1 1"), rearrange(vector_tmp, "c -> 1 c"))
        # iloss2 = - torch.log(vector_sim2.exp() / cos_map2.exp().sum())

        ### extrapolate
        # vector_tmp2 = vector_tmp * ratio2 + vector_feature * (1-ratio2)
        cos_map3 = cosfn(rearrange(afeature_map, "c h w -> 1 c h w"), rearrange(vector_tmp, "c -> 1 c 1 1")).squeeze()
        cos_map3 = torch.clamp(cos_map3, 0., 1.)
        # print(cos_map3.shape, gt_h, gt_w)
        cos_map3 = cos_map3.exp()
        gt_value = cos_map3[gt_h[bi], gt_w[bi]]
        cos_map3 = cos_map3[min_x:max_x, min_y:max_y]
        iloss = - torch.log(gt_value / cos_map3.sum())
        gt_values_to_record.append(gt_value.clone().detach().log().cpu())
        total_loss.append(iloss)
    gt_values_to_record = torch.stack(gt_values_to_record).mean()
    return torch.stack(total_loss).mean(), gt_values_to_record


def _tmp_loss_func(layer_i, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-layer_i)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[layer_i][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[layer_i], tmpl_feature)  # shape [8,12,12]
    loss, gtv = extrapolate_loss(raw_fea_list[layer_i], tmpl_feature, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss, gtv


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

        nearby = self.config['special']['nearby']
        loss_0, gt_value_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
        loss_1, gt_value_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_2, gt_value_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_3, gt_value_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_4, gt_value_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)

        # import ipdb; ipdb.set_trace()
        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

        loss_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3,
                     'loss_4': loss_4}
        gtvs_dict = {'gtv_0': gt_value_0, 'gtv_1': gt_value_1, 'gtv_2': gt_value_2, 'gtv_3': gt_value_3,
                     'gtv_4': gt_value_4}
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



def train(logger, config):
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    # ------------------------
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id,
                    collect_sim=False, split="Test1+2", upsample="bilinear")
    monitor = Monitor(key='mre', mode='dec')

    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list_list = []
    for i in range(len(testset)):
        landmark_list = testset.ref_landmarks(i)
        landmark_list_list.append(landmark_list)
    config_spe = config['special']
    dataset_train = Cephalometric(config['dataset']['pth'],
                                  patch_size=config['special']['patch_size'],
                                  entr_map_dir=config['dataset']['entr'],
                                  prob_map_dir=config['dataset']['prob'],
                                  mode="Train", use_prob=True, pre_crop=False, retfunc=2,
                                  cj_brightness=config_spe['cj_brightness'],
                                  cj_contrast=config_spe['cj_contrast'],
                                  cj_saturation=config_spe['cj_saturation'],
                                  cj_hue=config_spe['cj_hue'],
                                  sharpness=0.2)
    dataset_train.prob_map_for_all(landmark_list_list)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


def test(logger, config):
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")


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
