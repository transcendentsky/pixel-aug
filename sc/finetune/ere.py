import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from datasets.ceph.ceph_heatmap import Cephalometric
from utils.tester.tester_heatmap import Tester
# from models.unetr import UNETR
from models.network import UNet_Pretrained
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from einops import repeat, rearrange


EX_CONFIG = {
    "special": {
        "pretrain_model": "/home1/quanquan/code/landmark/code/runs/ssl/ssl_pos_ip/debug/ckpt_v/model_best.pth",
    },
    "training": {
        "load_pretrain_model": True,
    },
}

def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)


def nll_across_batch(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=(2, 3)))

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
        self.net = UNet_Pretrained(3, 19, regression=False, two_d_softmax=True)
        # self.net = UNETR(in_channels=3, n_classes=19, regression=False)
        self.loss_logic_fn = focal_loss
        self.loss_regression_fn = L1Loss
        self.config_training = config['training']
        # self.bce_loss_fn = torch.nn.BCELoss()
        self.bcewithlogits_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, **kwargs):
        return self.net.forward(x, no_sigmoid=True)

    def _tmp_delta(self, sigmoid):
        assert sigmoid.shape == (8,19,384,384), f"Got {sigmoid.shape}"
        h, w = sigmoid.shape[-2], sigmoid.shape[-1]
        _sum = sigmoid.sum(axis=-1)
        _sum = _sum.sum(axis=-1)
        _sum = repeat(sigmoid, "b c -> b c h w", h=384, w=384)
        sigmoid /= _sum
        return sigmoid

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        heatmap = data['heatmap']

        output = self.net.forward(img)
        # _sigmoid = self._tmp_delta(heatmap_pred)
        loss = nll_across_batch(output, heatmap)
        # logic_loss = self.bcewithlogits_fn(heatmap_pred, heatmap)
        # loss = logic_loss

        if torch.isnan(loss):
            print("debug: ", img.shape, output.shape, loss)
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
        state_dict = torch.load(self.config['special']['pretrain_model'])
        model_dict = self.net.state_dict()
        # new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            if k in state_dict.keys():
                model_dict.update({k:state_dict[k]})
                print("load ", k)
        self.net.load_state_dict(model_dict)


def train(logger, config):
    cudnn.benchmark = True
    assert logger is not None

    model = Learner(logger=None, config=config)
    tester = Tester(logger=None, config=config, mode="Test1+2")
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], ret_mode="onehot_heatmap")
    # trainer = DDPTrainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer.fit(model, dataset)


def test(logger, config):
    import os
    model = Learner(logger=None, config=config)
    tester = Tester(logger=logger, config=config, mode="Test1+2") # Test1+2
    # ckpt = tfilename(config['base']['runs_dir'], "ckpt/best_model_epoch_100.pth")
    ckpt = tfilename(config['base']['runs_dir']) + '/ckpt/model_latest.pth'
    # assert os.path.exists(ckpt), "???????????, "

    model.load(ckpt)
    model.cuda()
    model.eval()
    test_d = tester.test(model, draw=True)
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

