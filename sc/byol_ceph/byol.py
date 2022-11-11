"""
    Pixel Contrastive
"""

import torch
from models.byol import BYOL
from models.network_emb_study import UNet_Pretrained
import argparse
from datasets.ceph.ceph_ssl import Cephalometric
from torch import optim
from tutils import trans_init, trans_args, tfilename, dump_yaml, timer, save_script
import numpy as np
# from networks.barlowtwins import Learner
from tutils.trainer import Trainer, LearnerModule, DDPTrainer, Monitor, LearnerWrapper
import math
# from torch.cuda.amp import autocast, GradScaler
import os
from collections import OrderedDict
from sc.baseline.baseline_heatmap_spec import EX_CONFIG
from utils.tester.tester_byol import Tester

EX_CONFIG = {
    "special": {
        "patch_size": 256,
    }
}

class Learner(LearnerWrapper):
    def __init__(self, config, logger, **kwargs):
        super().__init__()
        self.config = config
        self.logger = logger
        self.bs = config['training']['batch_size']
        assert config['special']['non_local'] == True
        unet = UNet_Pretrained(n_channels=3, emb_len=config['special']['emb_len'], non_local=config['special']['non_local'])
        self.unet = unet
        self.net = BYOL(unet, image_size=config['dataset']['patch_size'], forward_func=4, projection_size=config['special']['emb_len'])
        self.msefn = torch.nn.MSELoss()
        self.cosfn = torch.nn.CosineSimilarity(dim=1)

    def forward(self, img, **kwargs):
        # For inference
        return self.net.module.net(img)

    def training_step(self, data, batch_idx, **kwargs):
        # with autocast():
        if True:
            img1 = data['raw_imgs'].float()
            img2 = data['crop_imgs'].float()
            p1 = data['raw_loc']
            p2 = data['chosen_loc']
            # print("training step: debug ", img1.shape, p1.shape, p2.shape)

            online_pred_one, online_pred_two, target_proj_one, target_proj_two, online_proj_one, online_proj_two = \
                self.net([img1, img2, p1, p2])

            loss_list = []
            log_dict = {}
            for i in range(len(online_pred_one)):
                loss1 = self.msefn(online_pred_one[i].squeeze(), target_proj_two[i].squeeze())
                loss2 = self.msefn(online_pred_two[i].squeeze(), target_proj_one[i].squeeze())
                loss = loss1 + loss2
                loss_list.append(loss)

                sim1 = self.cosfn(target_proj_one[i].squeeze(),
                                  target_proj_two[i].squeeze()).mean()
                sim2 = self.cosfn(online_proj_one[i].squeeze(),
                                  online_proj_two[i].squeeze()).mean()
                sim3 = self.cosfn(target_proj_one[i].squeeze(),
                                  online_pred_one[i].squeeze()).mean()
                sim4 = self.cosfn(target_proj_two[i].squeeze(),
                                  online_pred_two[i].squeeze()).mean()
                sim5 = self.cosfn(target_proj_one[i].squeeze(),
                                  online_pred_two[i].squeeze()).mean()
                sim6 = self.cosfn(target_proj_two[i].squeeze(),
                                  online_pred_one[i].squeeze()).mean()
                logitems = {f"l{i}_sim1": sim1, f"l{i}_sim2": sim2, f"l{i}_sim3": sim3, f"l{i}_sim4": sim4, f"l{i}_sim5": sim5, f"l{i}_sim6": sim6}
                log_dict = {**log_dict, **logitems}

            loss_total = torch.stack(loss_list).mean()
            loss_dict = {"loss": loss_total, "loss1": loss_list[0], "loss2": loss_list[1], "loss3": loss_list[2], "loss4": loss_list[3], "loss5": loss_list[4]}

            return {**loss_dict, **log_dict}

    def configure_optimizers(self):
        config_train = config['training']
        optimizer = optim.Adam(params=self.net.parameters(), \
                               lr=self.config['training']['lr'], betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=self.config['training']['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config['training']['decay_step'], gamma=config['training']['decay_gamma'])
        return {"optimizer": optimizer, "scheduler":scheduler}

    def on_before_zero_grad(self, **kwargs):
        """
            Called after optimizer.step() and before optimizer.zero_grad().
        """
        if self.net.module.use_momentum:
            self.net.module.update_moving_average()

    def load(self):
        # state_dict = torch.load(self.config['training']['pretrain_model'])
        # state_dict = '/home1/quanquan/code/landmark/code/runs/ssl/ssl/debug2/ckpt/best_model_epoch_890.pth'
        state_dict = torch.load(state_dict)
        self.unet.load_state_dict(state_dict)
        pass

def train(logger, config):
    tester = Tester(logger=None, config=config)
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], patch_size=config['special']['patch_size'])
    # dataset2 = DatasetTraining(config, mode="two_images")
    model = Learner(config=config, logger=logger)
    trainer = Trainer(config=config, logger=logger, tester=tester, monitor=monitor)
    trainer.fit(model, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/byol_ceph/byol.yaml")
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--func", default="train")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    # save_script(config['base']['runs_dir'], __file__)
    eval(args.func)(logger, config)
    # train()
    # test_v7()