"""
    Focus on a specific landmark, (not all landmarks)
"""
"""
    115.bmp: lm 0: 2.17 2.56, 0.52 0.36     |  0.588 2.511 4.321
            lm 3:  1.84 2.12, 0.55 0.40     |  0.415 2.113 4.133
            lm 7:  2.11 2.48, 0.49 0.32     |  0.690 1.943 4.070
            lm 17: 1.72 2.00, 0.40 0.20     |  0.918 3.224 5.017
            
    # averaged
    115.bmp: lm 0: 2.00 2.34, 0.55 0.40     |  0.588 2.511 4.321
            lm 3:  2.55 3.06, 0.47 0.29     |  0.415 2.113 4.133
            lm 7:  2.09 2.45, 0.46 0.29     |  0.690 1.943 4.070
            lm 17: 1.62 1.83, 0.54 0.40     |  0.918 3.224 5.017
"""

import torch
import numpy as np
import os
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from datasets.ceph.ceph_heatmap_spec import Cephalometric
from utils.tester.tester_heatmap_spec import Tester
# from models.unetr import UNETR
from models.network import UNet_Pretrained
import torch.backends.cudnn as cudnn
import random


EX_CONFIG={
    "special":{
        # 2.17 2.56, 0.52 0.36
        # "cj_brightness": [0.54, 1.62], # lm 17
        # "cj_contrast": [0.40, 1.83],
        # "cj_brightness": [0.55, 2.0], # lm 0
        # "cj_contrast": [0.40, 2.34],
        # "cj_brightness": [0.46, 2.09], # 7
        # "cj_contrast": [0.29, 2.45],
        "cj_brightness": 0.15, # any
        "cj_contrast": 0.25,
        # "cj_brightness": [0.569,1.55], # 17 *
        # "cj_contrast": [0.426, 1.73],
        "landmark_id": 17,
    },
    "training":{
        "val_check_interval": 2,
    }
}

def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)

def get_config():
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default=None, help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, no_logger=True)
    return config


def focal_loss(pred, gt):
    return (-(1 - pred) * gt * torch.log(pred+1e-8) - pred * (1 - gt) * torch.log(1 - pred + 1e-8)).mean()


def L1Loss(pred, gt, mask=None):
    # L1 Loss for offset map
    assert (pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
    return distence.sum() / mask.sum()


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, 1, regression=False)
        self.loss_logic_fn = focal_loss
        self.loss_regression_fn = L1Loss
        self.config_training = config['training']
        # self.bce_loss_fn = torch.nn.BCELoss()
        self.bcewithlogits_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, **kwargs):
        return self.net.forward(x, no_sigmoid=True)

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        heatmap = data['heatmap']

        heatmap_pred = self.forward(img)
        logic_loss = self.bcewithlogits_fn(heatmap_pred, heatmap)
        loss = logic_loss

        if torch.isnan(loss):
            print("debug: ", img.shape, heatmap_pred.shape, loss)
            print("debug ", torch.isnan(heatmap))
        return {'loss': loss}

    def configure_optimizers(self, **kwargs):
        optimizer = torch.optim.Adam(params=self.net.parameters(),
                                      lr=self.config_training['lr'],
                                      betas=(0.9, 0.999),
                                      eps=1e-08,
                                      weight_decay=self.config_training['weight_decay'],
                                      )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                    self.config_training['decay_step'], \
                                                    gamma=self.config_training['decay_gamma'])
        return {'optimizer':optimizer, 'scheduler':scheduler}


def train(logger, config):
    cudnn.benchmark = True
    assert logger is not None

    model = Learner(logger=None, config=config)
    tester = Tester(logger=None, config=config, split="Test1+2", landmark_id=config['special']['landmark_id'])
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], ret_mode="heatmap_only", landmark_id=config['special']['landmark_id'])
    # trainer = DDPTrainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer.fit(model, dataset)


def test(logger, config):
    model = Learner(logger=None, config=config)
    tester = Tester(logger=logger, config=config, mode="Test1+2") # Test1+2
    ckpt = tfilename(config['base']['runs_dir']) + '/ckpt/model_latest.pth'
    model.load(ckpt)
    model.cuda()
    model.eval()
    test_d = tester.test(model, draw=True)
    logger.info(f"results: {test_d}")


if __name__ == '__main__':
    reproducibility(0)
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/baseline/heatmap.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    eval(args.func)(logger, config)

