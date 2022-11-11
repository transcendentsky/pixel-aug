"""
    Use Learner2 (network with UNet3DEncoder as backbone )
"""
import torch
import argparse
from tutils import trans_init, trans_args, tfilename, dump_yaml, save_script
import numpy as np
from tutils.trainer import Trainer, Monitor, LearnerModule, LearnerWrapper
import math
from datasets.ceph_ssl import Cephalometric
from utils.tester_ssl import Tester
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch import optim
from models.network_emb_study import UNet_Pretrained
from torch import nn
from models.barlowtwins_ceph import BarlowProjector, off_diagonal


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


class BarlowModel(nn.Module):
    def __init__(self, config):
        super(BarlowModel, self).__init__()
        self.net = UNet_Pretrained(3, emb_len=config['training']['emb_len'], non_local=config['training']['non_local'])
        self.projector = BarlowProjector(in_channel=config['training']['emb_len'])
        # self.bn = nn.BatchNorm1d(4096, affine=False)

    def forward(self, x, train=False, **kwargs):
        # For Inference
        if train:
            return
        else:
            return self.net(x)


class Learner(LearnerWrapper):
    def __init__(self, config, logger=None, is_train=True,**kwargs):
        super().__init__()
        self.net = BarlowModel(config)
        self.batch_size = config['training']['batch_size']
        self.config = config
        self.logger = logger
        self.lambd = 0.0051

        self.cosfn = nn.CosineSimilarity(dim=1)
        self.is_train = is_train

    def forward(self, x, **kwargs):
        # For Inference
        return self.net(x)

    def _tmp_loss_func(self, y1_list, y2_list, p1, p2, ii):
        scale = 2 ** (5-ii)
        bs = p1.shape[0]
        f_1 = torch.stack([y1_list[ii][[id], :, p1[id, 0]//scale, p1[id, 1]//scale] for id in range(bs)]).squeeze()
        f_2 = torch.stack([y2_list[ii][[id], :, p2[id, 0]//scale, p2[id, 1]//scale] for id in range(bs)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss = self.loss_fn(z1, z2)
        sim  = self.cosfn(f_1, f_2).mean()
        return loss, sim

    def training_step(self, data, batch_idx, **kwargs):
        img1 = data["raw_imgs"]
        img2 = data["crop_imgs"]
        p1 = data['raw_loc']
        p2 = data['chosen_loc']

        y1_list, y2_list = self.net(img1), self.net(img2)
        loss0, sim0 = self._tmp_loss_func(y1_list, y2_list, p1, p2, ii=0)
        loss1, sim1 = self._tmp_loss_func(y1_list, y2_list, p1, p2, ii=1)
        loss2, sim2 = self._tmp_loss_func(y1_list, y2_list, p1, p2, ii=2)
        loss3, sim3 = self._tmp_loss_func(y1_list, y2_list, p1, p2, ii=3)
        loss4, sim4 = self._tmp_loss_func(y1_list, y2_list, p1, p2, ii=4)

        loss = loss1 + loss2 + loss3 + loss4

        return {'loss': loss, "loss0": loss0, "loss1": loss1, "loss2": loss2, "loss3": loss3, "loss4": loss4,
                "sim0": sim0, "sim1": sim1, "sim2": sim2, "sim3": sim3, "sim4": sim4}

    def loss_fn(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        # torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def load(self):
        # state_dict = "/home1/quanquan/code/landmark/code/runs/byol-2d/byol_std/debug/ckpt/best_model_epoch_150.pth"
        # state_dict = torch.load(state_dict)
        # self.net.load_state_dict(state_dict)
        pass

    def configure_optimizers(self, **kwargs):
        config = self.config
        optimizer = optim.Adam(params=self.net.parameters(), \
                               lr=self.config['training']['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=self.config['training']['weight_decay'])

        scheduler = optim.lr_scheduler.StepLR(optimizer, config['training']['decay_step'], gamma=config['training']['decay_gamma'])
        return {'optimizer': optimizer, "scheduler": scheduler}


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/byol_ceph/barlowtwins.yaml")
    parser.add_argument("--pretrain", action='store_true')
    args = trans_args(parser)
    logger, config = trans_init(args)
    save_script(config['base']['runs_dir'], __file__)
    monitor = Monitor(key="mre", mode="dec")
    dataset = Cephalometric(config['dataset']['pth'], patch_size=config['training']['patch_size'])
    tester = Tester(logger, config)

    model = Learner(config=config, logger=logger)
    trainer = Trainer(config=config, logger=logger, monitor=monitor, tester=tester)
    trainer.fit(model, dataset)


if __name__ == '__main__':
    train()
