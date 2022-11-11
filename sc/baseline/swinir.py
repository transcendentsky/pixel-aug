import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from models.network import UNet_Pretrained
from datasets.ceph.ceph_basic import Cephalometric
from utils.tester.tester_baseline2 import Tester
from models.unetr import UNETR
from models.transformer.swin_ir import SwinIR


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
        # self.net = UNet_Pretrained(19, regression=True)
        img_size = (384, 384)
        self.net = SwinIR(upscale=1, img_size=img_size,
                          window_size=8, img_range=1., depths=[6, 6, 6, 6],
                          embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
        self.loss_logic_fn = focal_loss
        self.loss_regression_fn = L1Loss
        self.config_training = config['training']

    def forward(self, x, **kwargs):
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        mask = data['mask']
        offset_y = data['offset_y']
        offset_x = data['offset_x']

        heatmap, regression_y, regression_x = self.forward(img)

        logic_loss = self.loss_logic_fn(heatmap, mask)
        regression_loss_x = self.loss_regression_fn(regression_x, offset_x, mask)
        regression_loss_y = self.loss_regression_fn(regression_y, offset_y, mask)

        loss = regression_loss_x + regression_loss_y + logic_loss * self.config['training']['lambda']

        if torch.isnan(loss):
            print("debug: ", img.shape, mask.shape, heatmap.shape, loss)
        return {'loss': loss, "logic_loss": logic_loss, "regloss_x": regression_loss_x, "regloss_y":regression_loss_y}

    def configure_optimizers(self, **kwargs):
        optimizer = torch.optim.AdamW(params=self.net.parameters(),
                                      lr=self.config_training['lr'],
                                      betas=(0.9, 0.999),
                                      eps=1e-08,
                                      weight_decay=self.config_training['weight_decay'],
                                      )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                    self.config_training['decay_step'], \
                                                    gamma=self.config_training['decay_gamma'])
        return {'optimizer':optimizer, 'scheduler':scheduler}

    # def load(self, pth=None, *args, **kwargs):
    #     self.net.load_state_dict(torch.load(pth))


def train(logger, config):
    model = Learner(logger=None, config=config)
    tester_subtest = Tester(logger=None, config=config, mode="subtest")
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], return_offset=True)
    # trainer = DDPTrainer(logger=logger, config=config, tester=tester_subtest, monitor=monitor)
    trainer = Trainer(logger=logger, config=config, tester=tester_subtest, monitor=monitor)
    trainer.fit(model, dataset)


def test(logger, config):
    model = Learner(logger=None, config=config)
    tester = Tester(logger=logger, config=config, mode="Train") # Test1+2
    ckpt = tfilename(config['base']['runs_dir'], "ckpt/best_model_epoch_600.pth")
    model.load(ckpt)
    model.cuda()
    model.eval()
    test_d = tester.test(model)
    logger.info(f"results: {test_d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/baseline/baseline_reg.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    save_script(config['base']['runs_dir'], __file__)
    # dump_yaml(logger, config)
    eval(args.func)(logger, config)

