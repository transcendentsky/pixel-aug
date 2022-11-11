"""
    Shifted offsets

    Related work
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerWrapper, LearnerModule
import argparse
from torch import optim
from utils.tester_ssl import Tester
from datasets.ceph_ssl import Cephalometric
from models.network_emb_study import UNet_Pretrained

torch.backends.cudnn.benchmark = True



class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(LearnerModule, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config['training']['non_local'], emb_len=16)
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

        nearby = self.config['training']['nearby']
        loss_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
        loss_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

        return {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4}

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
    tester = Tester(logger, config)
    monitor = Monitor(key='mre', mode='dec')
    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['training']['patch_size'])
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    # print(learner.net)
    # print(learner.net_patch)

    # example_input = torch.randn(1, 3, 224, 224)
    # print(count_model(learner.net, example_input))
    # print(count_model(learner.net_patch, example_input))
    # print(count_model(learner, example_input))
    # import ipdb; ipdb.set_trace()
    trainer.fit(learner, dataset_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ssl/ssl.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args)
    save_script(config['base']['runs_dir'], __file__)
    print_dict(config)

    eval(args.func)(logger, config)
