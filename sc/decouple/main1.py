import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

from models.network import UNet, UNet_Pretrained
from datasets.ceph_basic import Cephalometric
from utils.tester_baseline2 import Tester
from PIL import Image
import numpy as np

from tutils import trans_args, trans_init, save_image, dump_yaml, load_yaml, tfilename, print_dict, CSVLogger, MetricLogger


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--config", default='configs/baseline/baseline_reg.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    args = trans_args(parser)
    logger, config = trans_init(args)
    args = argparse.Namespace(**config['base'])
    dump_yaml(logger, config)
    # print_dict(config)
    runs_dir = config['base']['runs_dir']
    train_indices = str(config['base']['indices']).split(',')
    train_indices = [int(ind) for ind in train_indices]

    net = UNet_Pretrained(3, config['num_landmarks'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    start_epoch = 0
    config_train = config['training']
    optimizer = optim.Adam(params=net.parameters(), \
                           lr=config_train['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])

    loss_logic_fn = focal_loss
    loss_regression_fn = L1Loss

    # Tester
    tester = Tester(logger, config, tag=args.tag)

    dataset = Cephalometric(config['dataset']['pth'], mode=args.data)
    sampler = torch.utils.data.sampler.SubsetRandomSampler(np.array(train_indices))
    dataloader = DataLoader(dataset, batch_size=config_train['batch_size'],
                            num_workers=config_train['num_workers'], sampler=sampler)
    csvlogger = CSVLogger(tfilename(config['base']['runs_dir'], 'csv'), mode='a+')

    metriclogger = MetricLogger(logger=logger)

    for epoch in range(start_epoch, config_train['num_epochs']):
        logic_loss_list = list()
        net.train()

        for img, mask, offset_y, offset_x, landmark_list in metriclogger.log_every(dataloader, print_freq=1, header=None):
            img, mask, offset_y, offset_x = img.cuda(), \
                                            mask.cuda(), offset_y.cuda(), offset_x.cuda()

            heatmap, regression_y, regression_x = net(img)

            logic_loss = loss_logic_fn(heatmap, mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)

            loss = regression_loss_x + regression_loss_y + logic_loss * config_train['lambda']
            # loss = (loss * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logic_loss_list.append(loss.cpu().item())
        logger.info("Epoch {} Training logic loss {} ". \
                    format(epoch, sum(logic_loss_list) / dataset.__len__()))
        scheduler.step()

        if (epoch + 1) % config_train['save_seq'] == 0 and epoch > 198:
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
            net.eval()
            _d = tester.test(net, epoch=epoch)
            _d['indices'] = train_indices
            _d['record_time'] = datetime.now()
            _d['epoch'] = epoch
            _d['test_mode'] = args.data
            csvlogger.record(_d)
    # # Test
    _d = tester.test(net)
    _d['indices'] = train_indices
    _d['record_time'] = datetime.now()
    _d['epoch'] = epoch
    _d['test_mode'] = "Test1+2"
    csvlogger.record(_d)
