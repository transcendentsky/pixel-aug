"""
    Get accurate global feature, by interpolation
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_fined import Tester
from datasets.ceph.ceph_ssl import Cephalometric
from models.network_emb_study import UNet_Pretrained
from tutils import print_dict


torch.backends.cudnn.benchmark = True

def feature_syn(feature_map, h, w):
    # For moving the center from left corner to the 'real' center
    h = h - 0.5
    w = w - 0.5
    assert len(feature_map.shape) == 3, f"Got {feature_map.shape}"
    h_floor, h_ceil, h_ratio = torch.floor(h), torch.ceil(h), torch.ceil(h) - h
    h_floor, h_ceil = torch.clamp(h_floor, 0, feature_map.shape[1]-1).int(), torch.clamp(h_ceil, 0, feature_map.shape[1]-1).int()
    w_floor, w_ceil, w_ratio = torch.floor(w), torch.ceil(w), torch.ceil(w) - w
    w_floor, w_ceil = torch.clamp(w_floor, 0, feature_map.shape[2]-1).int(), torch.clamp(w_ceil, 0, feature_map.shape[2]-1).int()
    # import ipdb; ipdb.set_trace()
    fea1 = (feature_map[:, h_floor, w_floor] * h_ratio + feature_map[:, h_ceil, w_floor] + (1-h_ratio))
    fea2 = (feature_map[:, h_floor, w_ceil] * h_ratio + feature_map[:, h_ceil, w_ceil] + (1-h_ratio))
    fea = fea1 * w_ratio + fea2 * (1-w_ratio)
    return fea

cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, gt_values=None, nearby=None):
    b, w, h = cos_map.shape
    total_loss = list()
    assert gt_values.shape[0] == 8, f"Got {gt_values}"
    if torch.isnan(gt_values.any()):
        print(gt_values)
        import ipdb; ipdb.set_trace()
    for id in range(b):
        cos_map[id] = cos_map[id].exp()
        # gt_value = cos_map[id, gt_x[id], gt_y[id]].clone()
        gt_value = gt_values[id].exp()
        if nearby is not None:
            min_x, max_x = max(gt_x[id].int() - nearby, 0), min(gt_x[id].int() + nearby, w)
            min_y, max_y = max(gt_y[id].int() - nearby, 0), min(gt_y[id].int() + nearby, h)
            # import ipdb; ipdb.set_trace()
            chosen_patch = cos_map[id, min_x:max_x, min_y:max_y]
        else:
            chosen_patch = cos_map[id]
        id_loss = - torch.log(gt_value / (chosen_patch.sum() + gt_value))
        # if torch.isnan(id_loss):
        #     import ipdb; ipdb.set_trace()
        total_loss.append(id_loss)
    return torch.stack(total_loss).mean()


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc / scale, chosen_loc / scale
    tmpl_feature = torch.stack([feature_syn(crop_fea_list[ii][id, :, :, :], chosen_loc[id][0], chosen_loc[id][1])
                 for id in range(chosen_loc.shape[0])])
    raw_feature = torch.stack([feature_syn(raw_fea_list[ii][id, :, :, :], raw_loc[id][0], raw_loc[id][1])
                               for id in range(chosen_loc.shape[0])])
    product0 = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    gt_values = match_inner_product(raw_feature.unsqueeze(-1).unsqueeze(-1), tmpl_feature).squeeze()
    # print("_tmp_loss_func: gt_values", gt_values)
    # import ipdb; ipdb.set_trace()
    loss = ce_loss(product0, raw_loc[:, 0].int(), raw_loc[:, 1].int(), gt_values=gt_values, nearby=nearby)
    # import ipdb; ipdb.set_trace()
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

        nearby = self.config['special']['nearby']
        loss_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
        loss_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
        ret_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4}
        # print_dict(ret_dict)
        # import ipdb; ipdb.set_trace()
        # if torch.isnan(loss):
        #     pass
        return ret_dict

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
    tester = Tester(logger, config, default_oneshot_id=114, collect_sim=False)
    monitor = Monitor(key='mre', mode='dec')

    id_oneshot = 124
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list = testset.ref_landmarks(id_oneshot)
    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['special']['patch_size'],
                                  pre_crop=False, ref_landmark=landmark_list, use_prob=True, retfunc=2)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


def test(logger, config):
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")

    learner = Learner(logger=logger, config=config)
    learner.load(tfilename(config['base']['runs_dir'], 'ckpt', 'best_model_epoch_400.pth'))
    learner.cuda()
    learner.eval()

    # ids = [1,2,3,4,5,6,7,8,9]
    ids = [114, 124, 125, ]
    for id_oneshot in ids:
        res = tester.test(learner, oneshot_id=id_oneshot)
        logger.info(res)
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print(config['base'])

    eval(args.func)(logger, config)
