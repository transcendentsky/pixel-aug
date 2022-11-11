"""
    Positive pairs interpolation + U2net model for GU2Net paper
"""


import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_u2net import Tester
from datasets.ceph.ceph_ssl_adapm import Cephalometric
from models.u2net import U2Net
from einops import rearrange, repeat
import random
import numpy as np


torch.backends.cudnn.benchmark = True

# get from sift points comparison
EX_CONFIG = {
    "special": {
        'size': [384, 384],
        "cj_brightness": 1.6, # 0.15
        "cj_contrast": 1.2, # 0.25
        "cj_saturation": 0.3, # 0.
        "cj_hue": 0.2, # 0
        "weighted_t": 1/600, # temperature
        "pretrain_model": "/home1/quanquan/code/landmark/code/runs/ssl/ssl_pos_ip/debug/ckpt/model_epoch_500.pth"
    },
    "dataset": {
        'entr': '/home1/quanquan/datasets/Cephalometric/entr1/train/',
        'prob': '/home1/quanquan/datasets/Cephalometric/prob/train/',
    },
    # "training": {
    #     "load_pretrain_model": True,
    #     "lr": 0.0001,
    # }
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


Momented_GT_VALUE = None

def extrapolate_loss(feature, tmpl_feature, gt_h, gt_w, nearby, epoch=0, mgt=None):
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
        gt_values_to_record.append(gt_value.clone().detach().log())
        total_loss.append(iloss)
    gt_values_to_record = torch.stack(gt_values_to_record)
    total_loss = torch.stack(total_loss)
    # std_loss = total_loss.mean().detach()
    # if EX_CONFIG['special']['weighted_t'] > 0 and epoch >= 300:
    #     weight = _tmp_get_weight(gt_values_to_record, epoch=epoch)
    #     total_loss *= weight
    #     assert gt_values_to_record.mean() >= (gt_values_to_record * weight).mean(), f"Got {gt_values_to_record}, {weight}"
    gt_values_to_record = gt_values_to_record.mean()
    total_loss_mean = total_loss.mean()

    if False:
        # momented gt_values
        if mgt is None:
            mgt = gt_values_to_record
        mgt = mgt * 0.99 + gt_values_to_record * 0.01
        total_loss_mean *= ((mgt+1e-3) / (gt_values_to_record+1e-3)) ** 0.5 # temperature
    # assert total_loss_mean >= std_loss, f"Got {total_loss_mean}, {std_loss}"
    return total_loss_mean, gt_values_to_record.cpu()


def _tmp_loss_func(layer_i, raw_fea_list, raw_loc, crop_fea_list1, chosen_loc1, crop_fea_list2, chosen_loc2, nearby=None, epoch=0, mgt=None):
    scale = 2 ** (4-layer_i)
    raw_loc, chosen_loc1, chosen_loc2 = raw_loc // scale, chosen_loc1 // scale, chosen_loc2 // scale

    tmpl_feature1 = torch.stack([crop_fea_list1[layer_i][[id], :, chosen_loc1[id][0], chosen_loc1[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    tmpl_feature2 = torch.stack([crop_fea_list1[layer_i][[id], :, chosen_loc1[id][0], chosen_loc1[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)
    tmpl_feature = tmpl_feature1 * lam + tmpl_feature2 * (1-lam)
    # product = match_inner_product(raw_fea_list[layer_i], tmpl_feature)  # shape [8,12,12]
    loss, gtv = extrapolate_loss(raw_fea_list[layer_i], tmpl_feature, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby, epoch=epoch, mgt=mgt)
    return loss, gtv


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = U2Net(1, 19)
        self.net_patch = U2Net(1, 19)
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()
        self.momented_gt_value = [None for _ in range(5)]

    def forward(self, x, **kwargs):
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        raw_loc  = data['raw_loc']
        crop_imgs1 = data['crop_imgs1']
        chosen_loc1 = data['chosen_loc1']
        crop_imgs2 = data['crop_imgs2']
        chosen_loc2 = data['chosen_loc2']
        epoch = kwargs['epoch']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list1 = self.net_patch(crop_imgs1)
        crop_fea_list2 = self.net_patch(crop_imgs2)

        nearby = self.config['special']['nearby']
        loss_0, gt_value_0 = _tmp_loss_func(0, raw_fea_list, raw_loc, crop_fea_list1, chosen_loc1, crop_fea_list2, chosen_loc2, epoch=epoch, mgt=self.momented_gt_value[0])
        loss_1, gt_value_1 = _tmp_loss_func(1, raw_fea_list, raw_loc, crop_fea_list1, chosen_loc1, crop_fea_list2, chosen_loc2, nearby, epoch=epoch, mgt=self.momented_gt_value[1])
        loss_2, gt_value_2 = _tmp_loss_func(2, raw_fea_list, raw_loc, crop_fea_list1, chosen_loc1, crop_fea_list2, chosen_loc2, nearby, epoch=epoch, mgt=self.momented_gt_value[2])
        loss_3, gt_value_3 = _tmp_loss_func(3, raw_fea_list, raw_loc, crop_fea_list1, chosen_loc1, crop_fea_list2, chosen_loc2, nearby, epoch=epoch, mgt=self.momented_gt_value[3])
        loss_4, gt_value_4 = _tmp_loss_func(4, raw_fea_list, raw_loc, crop_fea_list1, chosen_loc1, crop_fea_list2, chosen_loc2, nearby, epoch=epoch, mgt=self.momented_gt_value[4])

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
            self.logger.info(f"Load Pretrain Model from config: {self.config['special']['pretrain_model']}")
            state_dict = torch.load(self.config['special']['pretrain_model'])
            self.net.load_state_dict(state_dict)
        else:
            self.logger.info(f"Load Pretrain Model: {pth}")
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
    size = config['special']['size']
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id,
                    collect_sim=False, split="Test1+2", upsample="bilinear", size=size)
    monitor = Monitor(key='mre', mode='dec')

    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list_list = []
    for i in range(len(testset)):
        landmark_list = testset.ref_landmarks(i)
        landmark_list_list.append(landmark_list)
    config_spe = config['special']
    dataset_train = Cephalometric(config['dataset']['pth'],
                                  size=size,
                                  patch_size=config['special']['patch_size'],
                                  entr_map_dir=config['dataset']['entr'],
                                  prob_map_dir=config['dataset']['prob'],
                                  mode="Train", use_prob=True, pre_crop=False, retfunc=2,
                                  cj_brightness=config_spe['cj_brightness'],
                                  cj_contrast=config_spe['cj_contrast'],
                                  cj_saturation=config_spe['cj_saturation'],
                                  cj_hue=config_spe['cj_hue'],
                                  sharpness=0.2)
    dataset_train.prob_map_for_all(landmark_list_list, size=size)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


def test(logger, config):
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ssl2/smp.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print_dict(config)
    # import ipdb; ipdb.set_trace()

    eval(args.func)(logger, config)
