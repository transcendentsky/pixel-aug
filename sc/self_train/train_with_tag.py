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

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config", default="configs/self_train/self_train.yaml", help="default configs")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--pseudo", type=str, default="")
    parser.add_argument("--oneshot", type=int, default=126)
    parser.add_argument('--finaltest', action='store_true')
    args = parser.parse_args()
    logger, config = trans_init(args, file=__file__)
    csvlogger = CSVLogger(config['base']['runs_dir'])
    config_train = config['training']

    # Settings
    # pseudo_pth = "/home1/quanquan/code/landmark/code/runs/ssl/dump_label_from_ssl/prob_3_id_124/pseudo_labels_init"
    pseudo_pth = "/home1/quanquan/code/landmark/code/runs/ssl/dump_label_from_ssl/ssl_neg_interp/pseudo_labels_init"
    oneshot_id = 114

    oneshot_dataset = Cephalometric(config['dataset']['pth'], 'Oneshot', oneshot_id=oneshot_id)
    oneshot_dataloader = DataLoader(oneshot_dataset, batch_size=1,
                                    shuffle=True, num_workers=config_train['num_workers'])

    net = UNet_Pretrained(3, config_train['num_landmarks'], regression=True)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    start_epoch = 0

    if args.epoch > 0:
        start_epoch = args.epoch + 1
        logger.info("Loading checkpoints from epoch {}".format(args.epoch))
        checkpoints = torch.load(os.path.join(config['base']['runs_dir'], \
                                              "model_epoch_{}.pth".format(args.epoch)))
        net.load_state_dict(checkpoints)

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

    for epoch in range(start_epoch, config_train['num_epochs']):
        logic_loss_list = list()
        net.train()

        select_epoch = epoch if epoch > 100 else 0
        dataset = Cephalometric(config['dataset']['pth'], mode='pseudo', pseudo_pth=pseudo_pth)
        dataloader = DataLoader(dataset, batch_size=config_train['batch_size'] - 1,
                                drop_last=True, shuffle=True, num_workers=config_train['num_workers'])
        ii = 0
        for data in tqdm(dataloader, ncols=100):
            img, mask, offset_x, offset_y, landmark_list = data['img'], data['mask'], data['offset_x'], data['offset_y'], data['landmark_list']
            oneshot_generater = iter(oneshot_dataloader)
            data2 = next(oneshot_generater)
            shot_img, shot_mask, shot_offset_x, shot_offset_y = data2['img'], data2['mask'], data2['offset_x'], data2['offset_y']

            img = torch.cat([img, shot_img], 0)
            mask = torch.cat([mask, shot_mask], 0)
            offset_y = torch.cat([offset_y, shot_offset_y], 0)
            offset_x = torch.cat([offset_x, shot_offset_x], 0)

            img, mask, offset_y, offset_x = img.cuda(), \
                                            mask.cuda(), offset_y.cuda(), offset_x.cuda()
            # import ipdb; ipdb.set_trace()

            heatmap, regression_y, regression_x = net(img)

            logic_loss = loss_logic_fn(heatmap, mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)

            loss = regression_loss_x + regression_loss_y + logic_loss * config_train['lambda']
            # import ipdb; ipdb.set_trace()

            # print(f"Epoch {epoch}:{ii} | loss: {loss}")
            ii += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logic_loss_list.append(loss.cpu().item())

        logger.info("Epoch {} Training logic loss {} ".\
                    format(epoch, sum(logic_loss_list) / dataset.__len__()))
        logger.info(f"learning_rate: {get_lr(optimizer)}")
        scheduler.step()
        # np.save("debug_data.npy", [img.cpu().detach().numpy(), mask.cpu().detach().numpy(), offset_x.cpu().detach().numpy(), offset_y.cpu().detach().numpy(), heatmap.cpu().detach().numpy(), regression_y.cpu().detach().numpy(), regression_x.cpu().detach().numpy()] )
        # torch.save([img, mask, offset_x, offset_y, landmark_list, heatmap, regression_y, regression_x], "debug_data.pt")
        # # save model
        if epoch == -1 or (epoch + 1) % config_train['save_seq'] == 0:
            logger.info(config['base']['runs_dir'] + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), config['base']['runs_dir'] + "/model_epoch_{}.pth".format(epoch))

            config_train['last_epoch'] = epoch
            net.eval()
            res = tester.test(net, epoch=epoch)
            mre = res['mre']
            if mre < best_mre:
                best_mre = mre
                best_epoch = epoch
                save_dict = {"epoch": best_epoch, "mre": best_mre,
                             "model": config['base']['runs_dir'] + "/model_best.pth".format(epoch),
                             # "model_patch": config['base']['runs_dir'] + "/model_patch_epoch_{}.pth".format(epoch),
                             }
                res['epoch'] = epoch
                csvlogger.record(res)
            logger.info(f"********  Best MRE:{best_mre} in Epoch {best_epoch} || Epoch{epoch}:{mre} ********")

        with open(config['base']['runs_dir'] + "/config.yaml", "w") as f:
            yaml.dump(config, f)
