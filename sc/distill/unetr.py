import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger, ProLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from models.network import UNet_Pretrained
from datasets.ceph.ceph_basic import Cephalometric
from utils.tester.tester_baseline2 import Tester
from models.unetr import UNETR
from torch.utils.data import DataLoader


def focal_loss(pred, gt):
    return (-(1 - pred) * gt * torch.log(pred+1e-8) - pred * (1 - gt) * torch.log(1 - pred + 1e-8)).mean()


def L1Loss(pred, gt, mask=None):
    # L1 Loss for offset map
    assert (pred.shape == gt.shape)
    gap = pred - gt
    distance = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distance = distance * mask
        return distance.sum() / mask.sum()
    else:
        return distance.mean()


def train(logger, config):
    # others
    tester = Tester(logger=None, config=config, mode="Test1+2")
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], return_offset=True)

    # __init__
    teacher = UNet_Pretrained(3, 19, regression=True)
    net = UNETR(in_channels=3, n_classes=19)
    loss_logic_fn = focal_loss
    loss_regression_fn = L1Loss
    config_training = config['training']
    state_dict = torch.load(
        '/home1/quanquan/code/landmark/code/runs/baseline/baseline_reg/run1/ckpt/best_model_epoch_600.pth')
    new_dict = { k[7:]:v for k, v in state_dict.items()}
    teacher.load_state_dict(new_dict)


    # configure_optimizers
    optimizer = torch.optim.AdamW(params=net.parameters(),
                                  lr=config_training['lr'],
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=config_training['weight_decay'],
                                  )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                config_training['decay_step'], \
                                                gamma=config_training['decay_gamma'])

    # trainer processing
    teacher.cuda()
    teacher.eval()
    net.cuda()
    net.train()
    trainloader = DataLoader(dataset=dataset,
                                  batch_size=config_training['batch_size'],
                                  num_workers=config_training['num_workers'],
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True)

    # training_step
    prologger = ProLogger(logger=logger, metric=("mre", "dec"), epoch=config_training['num_epochs'])
    for epoch, trainlogger, dtrain, monitor in prologger:
        for data in trainlogger(trainloader):
            img = data['img'].cuda()

            heatmap_0, regression_y_0, regression_x_0 = teacher(img)
            heatmap, regression_y, regression_x = net(img)

            logic_loss = loss_logic_fn(heatmap, heatmap_0)
            regression_loss_x = loss_regression_fn(regression_x, regression_x_0)
            regression_loss_y = loss_regression_fn(regression_y, regression_y_0)

            loss = regression_loss_x + regression_loss_y + logic_loss * config['training']['lambda']
            record_dict = {'loss': loss, "logic_loss": logic_loss, "regloss_x": regression_loss_x, "regloss_y": regression_loss_y}
            trainlogger.update(**record_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 50 == 2:
            dtest = tester.test(net)
            dbest = monitor.record(dtest, epoch=epoch)
            logger.info({**dbest, **dtest})

            torch.save(net.state_dict(), tfilename(config['base']['runs_dir'], "model_latest.pth"))
            if dbest['isbest']:
                torch.save(net.state_dict(), tfilename(config['base']['runs_dir'], f"best_model_epoch_{epoch}.pth"))


# def test(logger, config):
#     model = Learner(logger=None, config=config)
#     tester = Tester(logger=logger, config=config, mode="Test1+2") # Test1+2
#     ckpt = tfilename(config['base']['runs_dir'], "ckpt/best_model_epoch_600.pth")
#     model.load(ckpt)
#     model.cuda()
#     model.eval()
#     test_d = tester.test(model)
#     logger.info(f"results: {test_d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/baseline/baseline_unetr.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    save_script(config['base']['runs_dir'], __file__)
    # dump_yaml(logger, config)
    eval(args.func)(logger, config)

