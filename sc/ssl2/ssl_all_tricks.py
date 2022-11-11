"""
    tricks include
        ssl/interpolate.py    negative interpolate
        ssl2/fined2.py  fined cos map
        ssl2/ssl_probmap3.py  entropy map
        augment     ps-64
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
import numpy as np
from einops import repeat, rearrange
import random

torch.backends.cudnn.benchmark = True


USE_NOISE=False

def interpolate_gt_value(cos_map, h, w):
    # For moving the center from left corner to the 'real' center
    h = h - 0.5
    w = w - 0.5
    assert len(cos_map.shape) == 2, f"Got {cos_map.shape}, it should be [h, w]"
    h_floor, h_ceil, h_ratio = torch.floor(h), torch.ceil(h), torch.ceil(h) - h
    h_floor, h_ceil = torch.clamp(h_floor, 0, cos_map.shape[0]-1).int(), torch.clamp(h_ceil, 0, cos_map.shape[0]-1).int()
    w_floor, w_ceil, w_ratio = torch.floor(w), torch.ceil(w), torch.ceil(w) - w
    w_floor, w_ceil = torch.clamp(w_floor, 0, cos_map.shape[1]-1).int(), torch.clamp(w_ceil, 0, cos_map.shape[1]-1).int()
    if USE_NOISE:
        noise_h = np.random.random() * 0.5
        noise_w = np.random.random() * 0.5
        h_ratio = h_ratio + 0.2 * noise_h
        w_ratio = w_ratio + 0.2 * noise_w
    # import ipdb; ipdb.set_trace()
    v1 = (cos_map[h_floor, w_floor] * h_ratio + cos_map[h_ceil, w_floor] * (1-h_ratio))
    v2 = (cos_map[h_floor, w_ceil]  * h_ratio + cos_map[h_ceil, w_ceil]  * (1-h_ratio))
    gt_value = v1 * w_ratio + v2 * (1-w_ratio)
    assert not torch.isnan(gt_value)
    if gt_value > 1.0:
        print(f"", cos_map.max(), cos_map.min(), v1, v2, gt_value)
        import ipdb; ipdb.set_trace()
    assert gt_value <= 1.0, f"Got {gt_value}"
    return gt_value


cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, gt_values=None, nearby=None):
    bs, w, h = cos_map.shape
    total_loss = list()
    assert len(gt_values) == 8, f"Got {gt_values}"
    for i in range(bs):
        cos_map[i] = cos_map[i].exp()
        gt_value = gt_values[i].exp()
        if nearby is not None:
            min_x, max_x = max(gt_x[i].int() - nearby, 0), min(gt_x[i].int() + nearby, w)
            min_y, max_y = max(gt_y[i].int() - nearby, 0), min(gt_y[i].int() + nearby, h)
            chosen_patch = cos_map[i, min_x:max_x, min_y:max_y]
        else:
            chosen_patch = cos_map[i]
        id_loss = - torch.log(gt_value / (chosen_patch.sum() + gt_value))
        total_loss.append(id_loss)
    return torch.stack(total_loss).mean()


def _tmp_get_neg_loc(gt_h, gt_w, shape, nearby=None):
    h, w = shape
    while True:
        if nearby is not None:
            d_h = random.choice([1, 2, -1, -2, ])
            d_w = random.choice([1, 2, -1, -2, ])
        else:
            d_h = random.choice([1, -1])
            d_w = random.choice([1, -1])
        if gt_h+d_h >=0 and gt_h + d_h < h:
            if gt_w + d_w >= 0 and gt_w + d_w < w:
                break
    return d_h, d_w


def extrapolate_loss(feature, tmpl_feature, gt_h, gt_w, gt_values, nearby):
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)
    lam = lam if lam < 0.5 else 1-lam
    b, c, h, w = feature.shape
    ratio = 1.0 - lam
    ratio2 = 1.0 + lam

    b = tmpl_feature.shape[0]
    total_loss = []
    total_loss2 = []

    # float to int
    gt_h = gt_h.int()
    gt_w = gt_w.int()

    for bi in range(b):
        d_h, d_w = _tmp_get_neg_loc(gt_h[bi], gt_w[bi], shape=feature.shape[-2:], nearby=nearby)
        vector_tmp = tmpl_feature[bi]
        # vector_feature = feature[bi, :, gt_h[bi], gt_w[bi]]
        gt_value = gt_values[bi].exp()
        neg_vector = feature[bi, :, gt_h[bi]+d_h, gt_w[bi]+d_w]
        if nearby is not None:
            min_x, max_x = max(gt_h[bi] - nearby, 0), min(gt_h[bi] + nearby, w)
            min_y, max_y = max(gt_w[bi] - nearby, 0), min(gt_w[bi] + nearby, h)
        else:
            min_x, max_x = 0, h
            min_y, max_y = 0, w

        afeature_map = feature[bi]
        afeature_map = afeature_map * ratio + repeat(neg_vector, "c -> c h w", h=h, w=w) * (1-ratio)

        ### extrapolate
        cos_map3 = cosfn(rearrange(afeature_map, "c h w -> 1 c h w"), rearrange(vector_tmp, "c -> 1 c 1 1")).squeeze()
        cos_map3 = torch.clamp(cos_map3, 0., 1.)
        # print(cos_map3.shape, gt_h, gt_w)
        cos_map3 = cos_map3.exp()
        # gt_value = cos_map3[gt_h[bi], gt_w[bi]]
        cos_map3 = cos_map3[min_x:max_x, min_y:max_y]
        iloss = - torch.log(gt_value / cos_map3.sum())
        total_loss.append(iloss)
    return torch.stack(total_loss).mean()


def _tmp_loss_func(layer_i, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-layer_i)
    raw_loc = raw_loc / scale
    chosen_loc= chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[layer_i][[i], :, chosen_loc[i][0], chosen_loc[i][1]] \
                                for i in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[layer_i], tmpl_feature)  # shape [8,12,12]
    gt_values = [interpolate_gt_value(product[i], raw_loc[i, 0], raw_loc[i, 1]) for i in range(product.shape[0])]
    loss = extrapolate_loss(raw_fea_list[layer_i], tmpl_feature, raw_loc[:, 0], raw_loc[:, 1],
                            gt_values=gt_values, nearby=nearby)
    mean_gt_value = torch.stack(gt_values).mean().item()
    return loss, mean_gt_value

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
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id, collect_sim=False,
                    split="Test1+2", upsample="bilinear")
    monitor = Monitor(key='mre', mode='dec')

    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list_list = []
    for i in range(len(testset)):
        landmark_list = testset.ref_landmarks(i)
        landmark_list_list.append(landmark_list)

    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['special']['patch_size'],
                                  entr_map_dir=config['dataset']['entr'],
                                  prob_map_dir=config['dataset']['prob'],
                                  mode="Train", use_prob=True, pre_crop=False, retfunc=2)
    dataset_train.prob_map_for_all(landmark_list_list)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


def test(logger, config):
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print_dict(config)
    # import ipdb; ipdb.set_trace()

    eval(args.func)(logger, config)
