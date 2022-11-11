"""
    predict via Gaussion mask

    BUG!
        please switch to finetune3,

        reason for bug:
            focal_loss 
"""

import torch
import numpy as np
import argparse
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from models.network_for_finetune import UNet_finetune
from datasets.ceph.ceph_basic import Cephalometric
from utils.tester.tester import Tester
import random

from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils import torchvision_save


def focal_loss(pred, gt):
    return (-(1 - pred) * gt * torch.log(pred+1e-8) - pred * (1 - gt) * torch.log(1 - pred + 1e-8)).mean()


def reproducibility(seed=0):
    print("Reproducibility seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.autograd.set_detect_anomaly(True)


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_finetune(3, 19, regression=False, frozen_backbone=False, sigmoid_plus=True)
        self.loss_logic_fn = focal_loss
        self.config_training = config['training']

    def forward(self, x, **kwargs):
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        mask = data['mask']
        heatmap = self.forward(img)
        # import ipdb; ipdb.set_trace()
        loss = self.loss_logic_fn(heatmap, mask)
        if torch.isnan(loss):
            print("debug: ", img.shape, mask.shape, heatmap.shape, loss)
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

    def load(self, pth=None, *args, **kwargs):
        pth = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'
        print(f"Load pretrained model: {pth}")
        state_dict = torch.load(pth)
        # Load Partial Model
        model_dict = self.net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.net.load_state_dict(model_dict)


def train(logger, config):
    reproducibility(0)
    model = Learner(logger=None, config=config)
    model.load()
    tester_subtest = Tester(logger=None, config=config, mode="Test1+2")
    monitor = Monitor(key='mre', mode='dec')

    dataset = Cephalometric(config['dataset']['pth'], return_offset=False, use_gaussian_mask=True)
    # data = dataset.__getitem__(0)
    # mask = data['mask']
    # torchvision_save(mask[0], "debugmask.png")
    # import ipdb; ipdb.set_trace()


    trainer = Trainer(logger=logger, config=config, tester=tester_subtest, monitor=monitor)
    trainer.fit(model, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/ana/finetune.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # dump_yaml(logger, config)
    eval(args.func)(logger, config)

