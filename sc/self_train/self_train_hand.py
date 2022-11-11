import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor, Trainer
from models.network import UNet_Pretrained
# from datasets.hand_basic import HandXray
from utils.tester.tester_hand import Tester
from datasets.hand.hand_self_train import HandXray


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
        self.net = UNet_Pretrained(3, config['dataset']['n_cls'], regression=True)
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

        loss = regression_loss_x + regression_loss_y + logic_loss * self.config['special']['lambda']

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


def train(logger, config, args):
    model = Learner(logger=None, config=config)
    tester_subtest = Tester(logger=None, config=config, split="Test")
    monitor = Monitor(key='mre', mode='dec')
    # pseudo_path = "/home1/quanquan/code/landmark/code/runs/baseline/baseline_hand/" + args.pseudo
    # pseudo_path = tfilename(config['base']['runs_dir'], 'pseudo_labels_init')
    pseudo_path = args.pseudo
    dataset = HandXray(pathDataset=config['dataset']['pth'], pseudo_path=pseudo_path, split="pseudo", num_repeat=5)
    trainer = Trainer(logger=logger, config=config, tester=tester_subtest, monitor=monitor)
    trainer.fit(model, dataset)


def dump(logger, config, *args, **kwargs):
    """
    Dump pseudo labels from SSL-pretrained model
    """
    from tutils import dump_yaml
    from utils.tester.tester_hand_ssl import Tester as Tester_ssl
    from models.network_emb_study import UNet_Pretrained as UNet_SSL
    model = UNet_SSL(3, emb_len=config['special']['emb_len'], non_local=config['special']['non_local'])
    ckpt = config['training']['ckpt'] = config['special']['pseudo']
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)
    model.cuda()
    tester = Tester_ssl(logger, config, mode="Train")
    oneshot_id = config['special']['oneshot_id'] = 25
    _d, _ = tester.test_multi(model, oneshot_id=oneshot_id, dump_label=True)
    print(_d)
    dump_yaml(logger, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/self_train/self_train_hand.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--datanum", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    parser.add_argument("--pseudo", default='/home1/quanquan/code/landmark/code/runs/ssl/dump_label_from_ssl_hand/aug2/pseudo_labels_init')

    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    eval(args.func)(logger, config, args)

