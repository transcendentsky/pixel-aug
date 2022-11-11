import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
# from datasets.ceph.ceph_heatmap import Cephalometric
# from utils.tester.tester_heatmap import Tester
from models.network import UNet_Pretrained
from datasets.hand.hand_basic import HandXray
from utils.tester.tester_hand_heatmap import Tester
from tutils import torchvision_save, print_dict
import torch.backends.cudnn as cudnn


EX_CONFIG = {
    "special": {
        "pretrain_model": '/home1/quanquan/code/landmark/code/runs/ssl/ssl_hand/aug2/ckpt_v/model_best.pth',
    },
    "training":{
        "val_check_interval": 10,
        "load_pretrain_model": True,
}}


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
        self.net = UNet_Pretrained(3, 37, regression=False)
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
        # print(heatmap_pred.shape)
        # torchvision_save(img, tfilename("./.tmp/img_hand.png"))
        # torchvision_save(heatmap[:, 0, :, :].unsqueeze(1), tfilename("./.tmp/heatmap_hand.png"))
        # import ipdb; ipdb.set_trace()
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
        if pth is None:
            pth = self.config['special']['pretrain_model']
        pretrained_dict = torch.load(pth)
        model_dict = self.net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)


def train(logger, config):
    cudnn.benchmark = True
    assert logger is not None
    model = Learner(logger=None, config=config)
    tester = Tester(logger=None, config=config, split="Test")
    monitor = Monitor(key='mre', mode='dec')
    dataset = HandXray(config['dataset']['pth'], split="Train", ret_mode="heatmap_only", num_repeat=1)
    # trainer = DDPTrainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer.fit(model, dataset)


def test(logger, config):
    import os
    model = Learner(logger=None, config=config)
    tester = Tester(logger=logger, config=config, mode="Test") # Test1+2
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
    parser.add_argument("--config", default='configs/baseline/baseline_hand.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    print_dict(config)
    eval(args.func)(logger, config)

