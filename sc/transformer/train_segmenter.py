"""
    add regression
"""

import torch
import torchvision
import numpy as np
import argparse
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from tutils import tfilename, trans_args, trans_init, dump_yaml, tdir, save_script
from tutils.trainer import Trainer, LearnerModule, Monitor

from models.segmenter import Segmenter
from utils.eval import Evaluater
from utils.tester2 import Tester
from datasets.ceph_train import Cephalometric
from datasets.ceph_test import Test_Cephalometric
from models.loss import focal_loss, L1Loss

from torch.optim.lr_scheduler import StepLR
from models.scheduler import create_optimizer, create_scheduler

IGNORE_LABEL = 255


class SchedulerWrapper(object):
    def __init__(self, lr_scheduler):
        self.scheduler = lr_scheduler
        self.epoch = 0
        self.datalen = 150 * 4

    def get_lr(self):
        return self.scheduler.get_lr()

    def step(self):
        self.epoch += 1
        self.scheduler.step_update(num_updates=self.epoch * self.datalen)


class Learner(LearnerModule):
    def __init__(self, config, logger):
        super(Learner, self).__init__(config, logger)
        self.logger = logger
        self.net = Segmenter(config)
        self.loss_logic_fn = focal_loss
        self.loss_regression_fn = L1Loss
        self.lbda = config['special']['lambda']
        self.celoss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        self.config = config

    def forward(self, x, **kwargs):
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        img, mask, offset_y, offset_x, landmark_list = data['img'], data['mask'], data['offset_y'], data['offset_x'], data['landmark_list']
        heatmap, regression_x, regression_y = self.forward(img)

        logic_loss = self.loss_logic_fn(heatmap, mask)
        regression_loss_y = self.loss_regression_fn(regression_y, offset_y, mask)
        regression_loss_x = self.loss_regression_fn(regression_x, offset_x, mask)
        loss = regression_loss_x + regression_loss_y + logic_loss * self.lbda
        return {"loss": loss, "regress_loss_x": regression_loss_x, "regression_loss_y": regression_loss_y, "logic_loss": logic_loss}

    def configure_optimizers(self, **kwargs):
        # dataset_len = 150
        # scheduler_kwargs = dict(
        #     sched='polynomial',
        #     poly_step_size=1,
        #     iter_warmup=0.0,
        #     iter_max=dataset_len * self.config['training']['num_epochs'],
        #     poly_power=0.9,
        #     min_lr=1e-5,
        # )
        #
        # optimizer_kwargs = dict(
        #     opt='sgd',
        #     lr=self.config['optim']['lr'],
        #     weight_decay=self.config['optim']['weight_decay'],
        #     momentum=0.9,
        #     clip_grad=None,
        # )
        # optimizer = create_optimizer(argparse.Namespace(**optimizer_kwargs), self.net)
        optimizer = optim.AdamW(params=self.net.parameters(), lr=self.config['optim']['lr'],
                               weight_decay=self.config['optim']['weight_decay'])
        scheduler = StepLR(optimizer, self.config['training']['decay_step'], gamma=self.config['training']['decay_gamma'])
        # scheduler = create_scheduler(scheduler_kwargs, optimizer)
        # scheduler = SchedulerWrapper(scheduler)
        # scheduler = None
        return {'optimizer': optimizer, "scheduler": scheduler}

    def load(self, path=None):
        ckpt_path = self.config['network']['pretrain'] if path is None else path
        self.logger.info(f"Load Pretrain model `{ckpt_path}`")
        state_dict = torch.load(ckpt_path)
        self.net.load_state_dict(state_dict)


class FunctionManager(object):
    def __init__(self) -> None:
        super().__init__()

    def run_function(self, funcname, *args, **kwargs):
        getattr(self, funcname)(*args, **kwargs)

    def train(self, logger, config, args):
        # Moniter: key is the key in `Tester` to be detected
        monitor = Monitor(key='mre', mode='dec')
        # Tester: tester.test should return {"mre": mre, ...}
        tester_subtest = Tester(logger, config, mode="subtest")

        model = Learner(config, logger)
        # logger.info(f"Training with LIMITED samples")
        dataset = Cephalometric(config['dataset']['pth'])

        ######################  Trainer Settings  ###############################
        trainer = Trainer(logger, config, tester=tester_subtest, monitor=monitor,
                          tag=config['base']['tag'], runs_dir=config['base']['runs_dir'], val_check_interval=config['validation']['val_check_interval'], **config['training'])
        if args.pretrain:
            model.load()
        model.cuda()
        trainer.fit(model, dataset)

    def test(self, logger, config, args):
        model = Learner(config, logger)
        epoch = args.epoch
        pth = tfilename(config['runs_dir'], f"model_epoch_{epoch}.pth")
        model.load(pth)
        model.cuda()
        tester_train = Tester(logger, config, mode="Train")
        tester_test = Tester(logger, config, mode="Test1+2")
        logger.info(f"Dataset Training")
        tester_train.test(model)
        logger.info(f"Dataset Test 1+2")
        tester_test.test(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", default="train")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--config", default="configs/config_deit2.yaml")

    args = trans_args(parser)
    logger, config = trans_init(args)

    save_script(config['base']['runs_dir'], __file__)

    function_manager = FunctionManager()
    # getattr(function_manager, args.func)(logger, config, args)
    # eval(func)(*args)
    function_manager.run_function(args.func, logger, config, args)