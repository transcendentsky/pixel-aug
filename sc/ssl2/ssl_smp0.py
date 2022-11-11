"""
    Smp model for ERE paper / original ssl model without any tricks
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerWrapper, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_smp import Tester
from datasets.ceph.ceph_ssl import Cephalometric
from sc.ssl2.ssl_smp_model import Unet

torch.backends.cudnn.benchmark = True


EX_CONFIG = {
    "special": {
        "size": 384,
    },
    "training": {
        "val_check_interval": 10,
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
    #     total_loss.append(id_loss)
    # return torch.stack(total_loss).mean()


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    # loss = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    # return loss
    loss, gt_values = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss, gt_values


class Learner(LearnerWrapper):
    def __init__(self, logger, config, *args, **kwargs):
        super(LearnerWrapper, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = Unet()
        self.net_patch = Unet()
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.net.train()
        self.net_patch.train()

    def eval(self):
        self.net.eval()
        self.net_patch.eval()

    def cuda(self):
        self.net.cuda()
        self.net_patch.cuda()

    def to(self, rank):
        self.net.to(rank)
        self.net_patch.to(rank)

    def forward(self, x, **kwargs):
        # self.net(x['img'])
        # raise NotImplementedError
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        crop_imgs = data['crop_imgs']
        raw_loc  = data['raw_loc']
        chosen_loc = data['chosen_loc']
        # print(raw_imgs.shape, crop_imgs.shape)
        b, c, h, w = raw_imgs.shape
        # assert h == 800, f"Got {raw_imgs.shape}"

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list = self.net_patch(crop_imgs)

        nearby = self.config['special']['nearby']
        loss_0, gt_value_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
        loss_1, gt_value_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_2, gt_value_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_3, gt_value_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_4, gt_value_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

        loss_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4}
        gtvs_dict = {'gtv_0': gt_value_0, 'gtv_1': gt_value_1, 'gtv_2': gt_value_2, 'gtv_3': gt_value_3, 'gtv_4': gt_value_4}
        res_dict = {**loss_dict, **gtvs_dict}

        return res_dict

    def load(self):
        # state_dict = '/home1/quanquan/code/landmark/code/runs/ssl/ssl/baseline1/ckpt/best_model_epoch_50.pth'
        # state_dict = torch.load(state_dict)
        # self.net.load_state_dict(state_dict)
        pass

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
    template_oneshot_id = 114
    size = config['special']['size'] # [800,640]
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id,
                    collect_sim=False, split="Test1+2", upsample="bilinear", size=size)
    monitor = Monitor(key='mre', mode='dec')
    dataset_train = Cephalometric(pathDataset=config['dataset']['pth'], size=size, patch_size=config['special']['patch_size'])
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ssl2/smp.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    # save_script(config['base']['runs_dir'], __file__)
    print_dict(config)

    eval(args.func)(logger, config)
