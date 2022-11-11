from os import replace
import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor
from models.network import UNet_Pretrained
from datasets.ceph.ceph_basic import Cephalometric
from utils.tester.tester_baseline2 import Tester
# from tutils.trainer.trainer_abstract import AbstractTrainer
from torch.utils.data import DataLoader
from tutils.trainer import Trainer



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
        self.net = UNet_Pretrained(3, 19, regression=True)
        self.loss_logic_fn = focal_loss
        self.loss_regression_fn = L1Loss
        self.config_training = config['training']

    def forward(self, x, **kwargs):
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        mask = data['mask']
        offset_x = data['offset_x']
        offset_y = data['offset_y']

        heatmap, regression_y, regression_x = self.forward(img)

        logic_loss = self.loss_logic_fn(heatmap, mask)
        regression_loss_x = self.loss_regression_fn(regression_x, offset_x, mask)
        regression_loss_y = self.loss_regression_fn(regression_y, offset_y, mask)

        loss = regression_loss_x + regression_loss_y + logic_loss * self.config['training']['lambda']

        if torch.isnan(loss):
            print("debug: ", img.shape, mask.shape, heatmap.shape, loss)
        return {'loss': loss, "logic_loss": logic_loss, "regloss_x": regression_loss_x, "regloss_y":regression_loss_y}

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


class MyTrainer(Trainer):
    def __init__(self, logger, config, *args, **kwargs):
        self.select_indices = config['special']['indices']
        print("Chekc indices: ", self.select_indices)
        super(MyTrainer, self).__init__(logger, config, *args, **kwargs)

    def init_model(self, model, trainset, **kwargs):
        assert len(trainset) > 0 , f"Got {len(trainset)}"
        sampler = torch.utils.data.sampler.SubsetRandomSampler(np.array(self.select_indices))
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                num_workers=self.num_workers, sampler=sampler, pin_memory=True)
        if self.load_pretrain_model:
            model.load()
        model.net = torch.nn.DataParallel(model.net)
        model.cuda()

        return model


def train(logger, config):
    train_part_num = config['special']['num']
    if train_part_num == 0:
        train_indices = config['special']['indices']
        if isinstance(train_indices, str):
            train_indices = str(config['special']['indices']).split(',')
        elif isinstance(train_indices, list):
            train_indices = train_indices
            assert isinstance(train_indices[0], int)
        else:
            raise NotImplementedError
        config['special']['indices'] = [int(ind) for ind in train_indices]
    else:
        train_indices = np.random.choice(150,train_part_num, replace=False).tolist()
        config['special']['indices'] = train_indices

    logger.info(f"Len of indices: {len(config['special']['indices'])}")

    model = Learner(logger=None, config=config)
    tester_subtest = Tester(logger=None, config=config, mode="Test1+2")
    # tester_subtest = None
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], return_offset=True)
    trainer = MyTrainer(logger=logger, config=config, tester=tester_subtest, monitor=monitor)
    trainer.fit(model, dataset)

    tester_all = Tester(logger=logger, config=config, mode="Test1+2")
    res = tester_all.test(model, epoch=1)
    logger.info(res)


# 
def test(logger, config):
    csvlogger = CSVLogger(config['base']['runs_dir'])
    model = Learner(logger=None, config=config)
    tester_subtest = Tester(logger=None, config=config, mode="Test1+2")
    model.load(tfilename(config['base']['runs_dir'], "ckpt", "model_latest.pth"))
    model.cuda()
    model.eval()
    res = tester_subtest.test(model, epoch=1)
    csvlogger.record(res)
    print(res)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/baseline/baseline_reg.yaml', help="name of the run")
    parser.add_argument('--experiment', default='train_part')
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--num", type=int, default=0)
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)

    # config['training']['indices'] =
    eval(args.func)(logger, config)

