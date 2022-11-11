"""
    Pixel Contrastive
"""

import torch
from models.byol import BYOL
from models.network_emb_study import UNet_Pretrained
import argparse
# from datasets.ceph.ceph_ssl import Cephalometric
from torch import optim
from tutils import trans_init, trans_args, tfilename, dump_yaml, timer, save_script, print_dict
import numpy as np
# from networks.barlowtwins import Learner
from tutils.trainer import Trainer, LearnerModule, DDPTrainer, Monitor, LearnerWrapper
import math
# from torch.cuda.amp import autocast, GradScaler
import os
from collections import OrderedDict
# from utils.tester.tester_byol import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl import Test_Cephalometric


from datasets.ceph.ceph_ssl_adap import Cephalometric

EX_CONFIG = {
    "dataset": {
        'prob': '/home1/quanquan/datasets/Cephalometric/prob_pseudo/train/',
    },
    "special": {
        'patch_size': 256,
        # "cj_brightness": [0.538, 1.774], # avg
        # "cj_contrast": [0.384, 2.03],
        "cj_brightness": 0.15,
        "cj_contrast": 0.25,
        "use_adap_aug": True,
        'thresholds': [0, 2],
        
        "adap_params": {
            "le": {
                "cj_brightness": [0.6, 1.6], #[0.55, 1.7],	# [0.8, 1.4], 
                "cj_contrast":  [0.5, 1.7], # [0.4, 2.0], # [0.7, 1.4],   
                },
            "me": {
                "cj_brightness": [0.6, 1.6], # [0.55, 1.7],
                "cj_contrast":  [0.5, 1.7], # [0.4, 2.0],  
                },
            "he": {
                "cj_brightness": [0.8, 1.2],
                "cj_contrast": [0.8, 1.3],  
                },
        }
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

    def load(self, pth):
        # state_dict = torch.load(self.config['training']['pretrain_model'])
        pretrained_dict = torch.load(pth)
        # self.net.load_state_dict(state_dict)
        
        model_dict = self.net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        outside_dict = {k: v for k, v in pretrained_dict.items() if k not in model_dict}
        model_dict.update(pretrained_dict)
        # import ipdb; ipdb.set_trace()
        self.net.load_state_dict(model_dict)


def train(logger, config):
    tester = Tester(logger=None, config=config, upsample="nearest")
    monitor = Monitor(key='mre', mode='dec')
    # dataset = Cephalometric(config['dataset']['pth'], patch_size=config['training']['patch_size'])
    # dataset2 = DatasetTraining(config, mode="two_images")

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
                                  sharpness=0.2,
                                  use_adap_aug=config['special']['use_adap_aug'],
                                  runs_dir=config['base']['runs_dir'],
                                  adap_params=config['special']['adap_params'])
    # dataset_train.prob_map_for_all(landmark_list_list)
    dataset_train.entr_map_from_image3(thresholds=config['special']['thresholds'], inverse=True)
    dataset_train.entr_map_from_image()
    # dataset_train.entr_map_ushape(temperature=0.3)

    model = Learner(config=config, logger=logger)
    trainer = Trainer(config=config, logger=logger, tester=tester, monitor=monitor)
    trainer.fit(model, dataset_train)


def test(logger, config):
    tester = Tester(logger=None, config=config)
    learner = Learner(config=config, logger=logger)
    learner.load(tfilename(config['base']['runs_dir'], 'ckpt_v', 'model_latest.pth'))
    learner.cuda()
    learner.eval()
    res = tester.test(learner, oneshot_id=114, draw=True)
    logger.info(res)

    # ids = [1,2,3,4,5,6,7,8,9]
    # ids = [114, 124, 125, ]
    # for id_oneshot in ids:
    #     exit()



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