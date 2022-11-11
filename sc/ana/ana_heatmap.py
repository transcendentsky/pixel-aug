import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from datasets.ceph.ceph_heatmap import Cephalometric
from utils.tester.tester_heatmap import Tester
from models.network_emb_study import UNet_Pretrained as UNet_ssl
# from models.unetr import UNETR
from models.network import UNet_Pretrained
import torch.backends.cudnn as cudnn
import os
from tutils import print_dict



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
        ckpt = '/home1/quanquan/code/landmark/code/runs/baseline/baseline_part/part_150/ckpt/model_latest.pth'
        # ckpt = '/home1/quanquan/code/landmark/code/runs/baseline/baseline_heatmap/bce2/ckpt/model_latest.pth'
        self.net = UNet_Pretrained(3, 19, regression=True)
        self.load(ckpt)

    def forward(self, x, **kwargs):
        return self.net.forward(x, no_sigmoid=True)
        # self.net.get_concat_features(x, upsample="nearest")

    def get_features(self, x, *args, **kwargs):
        return self.net.get_features(x, *args, **kwargs)


class Learner2(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner2, self).__init__(*args, **kwargs)
        self.net = UNet_ssl(n_channels=3, emb_len=16, non_local=True)
        # "/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_3/ckpt/best_model_epoch_280.pth"
        ckpt = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'
        self.load(ckpt)

    def forward(self, x, **kwargs):
        # return self.net.forward(x, no_sigmoid=True)
        return self.net.get_concat_features(x, upsample="nearest")

    def get_features(self, x, *args, **kwargs):
        fea = self.net.get_concat_features(x, upsample="bilinear")
        # import ipdb; ipdb.set_trace()
        return fea

def ana(logger, config):
    from utils.tester.tester_heatmap_debug import Tester as Tester_debug
    model = Learner2(logger=None, config=config)
    # model = Learner(logger=None, config=config)

    tester = Tester_debug(logger=logger, config=config, split="Test1+2", collect_sim=True)

    model.cuda()
    model.eval()
    test_d = tester.test(model)
    # logger.info(f"results: {test_d}")
    print_dict(test_d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/baseline/heatmap.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # dump_yaml(logger, config)
    eval(args.func)(logger, config)

