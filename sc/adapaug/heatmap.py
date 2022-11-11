import os.path

import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
# from datasets.ceph.ceph_heatmap import Cephalometric
from datasets.ceph.ceph_heatmap_imaug import AdapColorJitter, Cephalometric
from utils.tester.tester_heatmap import Tester
# from models.unetr import UNETR
from models.network import UNet_Pretrained
from collections import OrderedDict
import torch.backends.cudnn as cudnn


EX_CONFIG = {
    "special": {
        # "pretrain_model": "/home1/quanquan/code/landmark/code/runs/ssl/ssl_pos_ip/debug/ckpt_v/model_best.pth",
        "pretrain_model": "/home1/quanquan/code/landmark/code/runs/ssl/ssl_mixup0/debug2/ckpt_v/model_best.pth",
    },
    "training": {
        "load_pretrain_model": True,
        # "use_amp": False,
    },
}


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
        self.net = UNet_Pretrained(3, 19, regression=False)
        # self.net = UNETR(in_channels=3, n_classes=19, regression=False)
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

    def load(self, pth=None, *args, **kwargs):
        # self.net.load_state_dict(torch.load(pth))
        assert os.path.exists(self.config['special']['pretrain_model']), f"Got ? {self.config['special']['pretrain_model']}"
        state_dict = torch.load(self.config['special']['pretrain_model'])
        model_dict = self.net.state_dict()
        # new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            if k in state_dict.keys():
                model_dict.update({k:state_dict[k]})
                print("load ", k)
        self.net.load_state_dict(model_dict)

    def load2(self, ckpt):
        self.net.load_state_dict(torch.load(ckpt))

def train(logger, config):
    cudnn.benchmark = True
    assert logger is not None

    model = Learner(logger=None, config=config)
    tester = Tester(logger=None, config=config, mode="Test1+2")
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], ret_mode="heatmap_only", adap_mode=2)
    trainer = DDPTrainer(logger=logger, config=config, tester=tester, monitor=monitor)
    # trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer.fit(model, dataset)


def test(logger, config):
    import os
    model = Learner(logger=None, config=config)
    tester = Tester(logger=logger, config=config, mode="Test1+2", get_mre_per_lm=True) # Test1+2
    # ckpt = tfilename(config['base']['runs_dir'], "ckpt/best_model_epoch_100.pth")
    ckpt = tfilename(config['base']['runs_dir']) + '/ckpt_v/model_best.pth'
    # assert os.path.exists(ckpt), "???????????, "

    model.load2(ckpt)
    model.cuda()
    model.eval()
    test_d = tester.test(model, draw=False)
    logger.info(f"results: {test_d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/finetune/config.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    eval(args.func)(logger, config)

