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

from models.network import UNet_Pretrained
from datasets.ceph.ceph_self_train import Cephalometric
from utils.tester.tester_baseline2 import Tester
from utils.utils import visualize


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config", default="configs/self_train/self_train.yaml", help="default configs")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--pseudo", type=str, default="")
    # parser.add_argument("--oneshot", type=int, default=126)
    parser.add_argument('--finaltest', action='store_true')
    args = parser.parse_args()
    logger, config = trans_init(args)
    csvlogger = CSVLogger(config['base']['runs_dir'])
    save_script(config['base']['runs_dir'], __file__)
    config_train = config['training']

    net = UNet_Pretrained(3, config_train['num_landmarks'], regression=True)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    logger.info("Loading checkpoints from epoch {}".format(args.epoch))
    # checkpoint = torch.load(os.path.join(config['base']['runs_dir'], \
    #                                       "model_epoch_{}.pth".format(args.epoch)))
    checkpoint_path = "/home1/quanquan/code/landmark/code/runs-st/train_part/prob_3_ref_10_scp/model_epoch_39.pth"
    print("Ckpt Path: ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint)

    tester_subtest = Tester(logger, config, mode="subtest")
    tester_train = Tester(logger, config, mode="Train")
    tester_test= Tester(logger, config, mode="Test1+2")  # Test1+2

    epoch = args.epoch
    logger.info(config['base']['runs_dir'] + "/model_epoch_{}.pth".format(epoch))
    torch.save(net.state_dict(), config['base']['runs_dir'] + "/model_epoch_{}.pth".format(epoch))

    config_train['last_epoch'] = epoch
    net.eval()
    # res = tester_subtest.test(net, epoch=9)
    # mre = res['mre']
    # logger.info(res)

    res = tester_train.test(net, epoch=9, draw=True)
    mre = res['mre']
    logger.info(res)

    # res = tester_test.test(net, epoch=9)
    # mre = res['mre']
    # logger.info(res)