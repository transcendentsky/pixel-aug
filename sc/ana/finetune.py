import argparse
import datetime
import os
from pathlib import Path
import time
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss
from PIL import Image
import numpy as np
from tutils import trans_init, trans_args, dump_yaml, tfuncname, tfilename, CSVLogger, save_script

from models.network_for_finetune import UNet_finetune
from datasets.ceph.ceph_self_train import Cephalometric
from utils.tester.tester_baseline2 import Tester
import random


def L1Loss(pred, gt, mask=None):
    # L1 Loss for offset map
    assert (pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
    return distence.sum() / mask.sum()


def gray_to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def focal_loss(pred, gt):
    return (-(1 - pred) * gt * torch.log(pred) - pred * (1 - gt) * torch.log(1 - pred)).mean()


def dump_best_config(logger, config, info):
    # dump yaml
    config = {**config, **info}
    with open(config['base']['runs_dir'] + "/best_config.yaml", "w") as f:
        yaml.dump(config, f)
    logger.info("Dump best config")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def reproducibility(seed=0):
    print("Reproducibility seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    # Parse command line options
    reproducibility(0)
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config", default="configs/ana/finetune.yaml", help="default configs")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--pseudo", type=str, default="")
    parser.add_argument("--oneshot", type=int, default=126)
    parser.add_argument('--finaltest', action='store_true')
    args = parser.parse_args()
    logger, config = trans_init(args, file=__file__)
    csvlogger = CSVLogger(config['base']['runs_dir'])
    config_train = config['training']

    net = UNet_finetune(3, config_train['num_landmarks'], regression=True, frozen_backbone=False)

    start_epoch = 0

    if True:
        pth = '/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap/prob_5_id_114/ckpt/best_model_epoch_400.pth'
        logger.info(f"Load pretrained model: {pth}")
        state_dict = torch.load(pth)
        # Load Partial Model
        model_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        net.load_state_dict(model_dict)

        # logger.info("Loading checkpoints from epoch {}".format(args.epoch))
        # state_dict = torch.load(os.path.join(config['base']['runs_dir'], \
        #                                       "model_epoch_{}.pth".format(args.epoch)))
        # net.load_state_dict(state_dict)

    net = torch.nn.DataParallel(net)
    net = net.cuda()

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=config_train['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])

    loss_logic_fn = focal_loss
    loss_regression_fn = L1Loss
    tester = Tester(logger, config, mode="Test1+2")  # Test1+2

    # Record
    best_mre = 100.0
    best_epoch = -1
    # ---------------
    # if True:
    #     for i in range(150):
    #         res = tester.test(net, epoch=epoch)

    dataset = Cephalometric(config['dataset']['pth'], mode='Train', pseudo_pth=None)
    dataloader = DataLoader(dataset, batch_size=config_train['batch_size'],
                            drop_last=True, shuffle=True, num_workers=config_train['num_workers'])

    for epoch in range(start_epoch, config_train['num_epochs']):
        logic_loss_list = list()
        net.train()

        select_epoch = epoch if epoch > 100 else 0
        ii = 0
        for data in tqdm(dataloader, ncols=100):
            img, mask, offset_x, offset_y, landmark_list = data['img'], data['mask'], data['offset_x'], data['offset_y'], data['landmark_list']
            img, mask, offset_y, offset_x = img.cuda(), mask.cuda(), offset_y.cuda(), offset_x.cuda()
            # import ipdb; ipdb.set_trace()
            heatmap, regression_y, regression_x = net(img)
            logic_loss = loss_logic_fn(heatmap, mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)

            loss = regression_loss_x + regression_loss_y + logic_loss * config_train['lambda']

            # print(f"Epoch {epoch}:{ii} | loss: {loss}")
            ii += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logic_loss_list.append(loss.cpu().item())
            if epoch == 0:
                print("check pineline!")
                break

        logger.info("Epoch {} Training logic loss {} ".\
                    format(epoch, sum(logic_loss_list) / dataset.__len__()))
        logger.info(f"learning_rate: {get_lr(optimizer)}")
        scheduler.step()

        # # save model
        if epoch == 0 or (epoch + 1) % config_train['save_seq'] == 0:
            logger.info(config['base']['runs_dir'] + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), config['base']['runs_dir'] + "/model_epoch_{}.pth".format(epoch))

            config_train['last_epoch'] = epoch
            net.eval()
            res = tester.test(net, epoch=epoch)
            mre = res['mre']
            if mre < best_mre:
                best_mre = mre
                best_epoch = epoch
                res['epoch'] = epoch
                csvlogger.record(res)
            logger.info(f"********  Best MRE:{best_mre} in Epoch {best_epoch} || Epoch {epoch}:{mre} ********")

