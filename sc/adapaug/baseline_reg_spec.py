"""
    Adaploss + regression
"""

import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from models.network import UNet_Pretrained
# from datasets.ceph.ceph_basic import Cephalometric
from datasets.ceph.ceph_reg_imaug import Cephalometric
from utils.tester.tester_baseline2 import Tester
from einops import reduce, rearrange, repeat


EX_CONFIG = {
    "special": {
        "use_adap_loss": False,
        # "aug_percent": [0.4525,0.3017,0.4369,0.9387,0.2926,0.4391,0.5921,0.6677,0.6218,0.3482,
                        # 0.2602,0.2478,0.1089,0.2296,0.1525,0.4434,0.2936,0.3175,0.6607],
        # "aug_percent": [1,0,0,0,0,0,0,0,0,0,
        #                 0,0,0,0,0,0,0,0,0,],
        # "cj_brightness": [0.54, 1.62], # lm 17
        # "cj_contrast": [0.40, 1.83],
        "cj_brightness": ((np.array([0.55, 2.0]) - 1) * 0.8 + 1).tolist(), # lm 0
        "cj_contrast": ((np.array([0.40, 2.34]) - 1) * 0.8 + 1).tolist(),
        # "cj_brightness": [0.46, 2.09], # 7
        # "cj_contrast": [0.29, 2.45],
        # "cj_brightness": 0.15, # any
        # "cj_contrast": 0.25,
        # "cj_brightness": [0.569,1.55], # 17 *
        # "cj_contrast": [0.426, 1.73],
        # "cj_brightness": 0.15, # old
        # "cj_contrast": 0.25,

        # "cj_brightness": 0.8,
        # "cj_contrast": 0.6,
        "lm_id": 0,
    },
    "training": {
        "load_pretrain_model": False,
        "val_check_interval": 50,
        "num_epochs": 201,
    },
}



def focal_loss(pred, gt):
    return (-(1 - pred) * gt * torch.log(pred+1e-8) - pred * (1 - gt) * torch.log(1 - pred + 1e-8))


def L1Loss(pred, gt, mask=None, ww=None):
    # L1 Loss for offset map
    assert (pred.shape == gt.shape)
    gap = pred - gt
    distance = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distance = distance * mask
    # import ipdb; ipdb.set_trace()

    distance = reduce(distance, "b c h w -> b c", reduction="sum")
    mask = reduce(mask, "b c h w -> b c", reduction="sum")
    res = distance / mask
    return res.mean()

class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, 1, regression=True)
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
        lm_id = config['special']['lm_id']
        # ww = data['ww']
        mask = mask[:, lm_id, :, :].unsqueeze(1)
        offset_x = offset_x[:, lm_id, :, :].unsqueeze(1)
        offset_y = offset_y[:, lm_id, :, :].unsqueeze(1)

        heatmap, regression_y, regression_x = self.forward(img)

        logic_loss = self.loss_logic_fn(heatmap, mask)
        logic_loss = reduce(logic_loss, "b c h w -> b c", reduction="sum")

        if self.config['special']['use_adap_loss']:
            raise NotImplementedError
        else:
            # logic_loss = logic_loss[:, lm_id]
            logic_loss = logic_loss.mean() / 10000.0
            regression_loss_x = self.loss_regression_fn(regression_x, offset_x, mask)
            regression_loss_y = self.loss_regression_fn(regression_y, offset_y, mask)

            loss = regression_loss_x + regression_loss_y + logic_loss * self.config['training']['lambda']
            # import ipdb; ipdb.set_trace()

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

    # def load(self, pth=None, *args, **kwargs):
    #     # assert os.path.exists(pth)
    #     print(f'Load CKPT from ``{pth}``')
    #     state_dict = torch.load(pth)
    #     self.load_state_dict(state_dict)

def train(logger, config):
    model = Learner(logger=None, config=config)
    tester_subtest = Tester(logger=None, config=config,
                            split="Test1+2",
                            get_mre_per_lm=True)
    lm_id = config['special']['lm_id']
    monitor = Monitor(key=f'mre_lm_{lm_id}', mode='dec')
    # dataset = Cephalometric(config['dataset']['pth'], return_offset=True)
    dataset = Cephalometric(config['dataset']['pth'],
                            adap_mode="none",
                            cj_brightness=config['special']['cj_brightness'],
                            cj_contrast=config['special']['cj_contrast'],
                            return_offset=True)
    trainer = Trainer(logger=logger, config=config, tester=tester_subtest, monitor=monitor)
    trainer.fit(model, dataset)


def test(logger, config):
    model = Learner(logger=None, config=config)
    tester = Tester(logger=logger, config=config, mode="Test1+2")
    ckpt = tfilename(config['base']['runs_dir'], "ckpt/best_model_epoch_600.pth")
    model.cuda()
    state_dict = torch.load(ckpt)
    new_dict = {}
    for k, v in state_dict.items():
        k2 = "net" + k[6:]
        new_dict[k2] = v
    model.load_state_dict(new_dict)
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
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    # save_script(config['base']['runs_dir'], __file__)
    # dump_yaml(logger, config)
    eval(args.func)(logger, config)

