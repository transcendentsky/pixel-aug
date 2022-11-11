"""
    finetune1 -> heatmap + regression
    finetune3 -> heatmap (BCE loss)

"""

import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from datasets.ceph.ceph_heatmap import Cephalometric
from utils.tester.tester_heatmap import Tester
# from models.unetr import UNETR
# from models.network import UNet_Pretrained
from models.network_for_finetune import UNet_finetune
from tutils import torchvision_save
import random


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
        # self.net = UNet_Pretrained(3, 19, regression=False)
        self.net = UNet_finetune(3, 19, regression=False, frozen_backbone=True)
        # self.net = UNETR(in_channels=3, n_classes=19, regression=False)
        self.loss_logic_fn = focal_loss
        self.loss_regression_fn = L1Loss
        self.config_training = config['training']
        self.bce_loss_fn = torch.nn.BCELoss()

    def forward(self, x, **kwargs):
        res = self.net(x)
        # torchvision_save(res[0].unsqueeze(1), "debug_heatmap.png")
        # import ipdb; ipdb.set_trace()
        return res

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        heatmap = data['heatmap']

        heatmap_pred = self.forward(img)
        logic_loss = self.bce_loss_fn(heatmap_pred, heatmap)
        loss = logic_loss
        # torchvision_save(heatmap[0].unsqueeze(1), "debug_heatmap_gt.png")
        # import ipdb; ipdb.set_trace()

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

    def load(self, pth=None, *args, **kwargs):
        if pth is None:
            pth = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'
        self.logger.info(f"Load pretrained model: {pth}")
        state_dict = torch.load(pth)
        # Load Partial Model
        model_dict = self.net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.net.load_state_dict(model_dict)

    # def load(self, pth=None, *args, **kwargs):
    #     self.net.load_state_dict(torch.load(pth))


def train(logger, config):
    reproducibility(0)
    model = Learner(logger=logger, config=config)
    learner = model
    learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl/test_qsyao/ckpt/best_model_epoch_100.pth')
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl/train1/ckpt/best_model_epoch_390.pth')
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl_ensemble/train2/ckpt/best_model_epoch_540.pth')
    # learner.load("/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth")
    # learner.load('/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth')
    # model.load()
    tester = Tester(logger=None, config=config, mode="Test1+2")
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], ret_mode="heatmap_only")
    # trainer = DDPTrainer(logger=logger, config=config, tester=tester_subtest, monitor=monitor)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer.fit(model, dataset)


def test(logger, config):
    import os
    model = Learner(logger=logger, config=config)
    tester = Tester(logger=logger, config=config, mode="Train") # Test1+2
    # ckpt = tfilename(config['base']['runs_dir'], "ckpt/best_model_epoch_100.pth")
    # ckpt = tfilename(config['base']['runs_dir']) + '/ckpt/model_latest.pth'
    # assert os.path.exists(ckpt), "???????????, "
    ckpt = '/home1/quanquan/code/landmark/code/runs/ana/finetune3/pretrain/ckpt_v/model_best.pth'

    model.load(ckpt)
    model.cuda()
    model.eval()
    test_d = tester.test(model, draw=True)
    logger.info(f"results: {test_d}")


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

