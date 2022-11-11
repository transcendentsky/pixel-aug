"""
    One source encoder +
    Two target encoder
        parallel training
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl_multi_encoder import Cephalometric
from models.network_emb_study import UNet_Pretrained


torch.backends.cudnn.benchmark = True


cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, nearby=None):
    b, w, h = cos_map.shape
    total_loss = list()
    gt_values_to_record = []
    for id in range(b):
        cos_map[id] = cos_map[id].exp()
        gt_value = cos_map[id, gt_x[id], gt_y[id]].clone()
        if nearby is not None:
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            chosen_patch = cos_map[id, min_x:max_x, min_y:max_y]
        else:
            chosen_patch = cos_map[id]
        id_loss = - torch.log(gt_value / chosen_patch.sum())
        gt_values_to_record.append(gt_value.clone().detach().log().cpu())
        total_loss.append(id_loss)
    gt_values_to_record = torch.stack(gt_values_to_record).mean()
    return torch.stack(total_loss).mean(), gt_values_to_record


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss, gt_values = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss, gt_values


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
        self.net_patch1 = UNet_Pretrained(3, emb_len=16)
        self.net_patch2 = UNet_Pretrained(3, emb_len=16)
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()
        self.patch_sizes = [64, 256]
        self.loss_gates = [1, 0, 0, 0, 0]
        self.logger.info(f"loss_gates: {self.loss_gates}, ps: {self.patch_sizes}")

    def forward(self, x, **kwargs):
        # self.net(x['img'])
        # raise NotImplementedError
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        raw_loc  = data['raw_loc']
        crop_imgs1 = data['crop_imgs1']
        chosen_loc1 = data['chosen_loc1']
        crop_imgs2 = data['crop_imgs2']
        chosen_loc2 = data['chosen_loc2']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list1 = self.net_patch1(crop_imgs1)
        crop_fea_list2 = self.net_patch2(crop_imgs2)

        nearby = self.config['special']['nearby']

        loss_dict = {}
        gt_dict = {}
        total_loss = []
        for i in range(5):
            crop_fea = crop_fea_list1 if self.loss_gates[i] == 0 else crop_fea_list2
            chosen_loc = chosen_loc1 if self.loss_gates[i] == 0 else chosen_loc2
            s_nearby = nearby if i > 0 else None
            loss, gt_value = _tmp_loss_func(i, raw_fea_list, crop_fea, raw_loc, chosen_loc, s_nearby)
            loss_dict[f"loss_{i}"] = loss
            gt_dict[f'gtv_{i}'] = gt_value
            total_loss.append(loss)

        total_loss = torch.stack(total_loss).sum()
        res_dict = {"loss": total_loss, **loss_dict, **gt_dict}
        return res_dict

    def load(self, pth=None, *args, **kwargs):
        if pth is None:
            print("Load Pretrain Model")
            state_dict = torch.load(self.config['network']['pretrain'])
            self.net.load_state_dict(state_dict)
        else:
            print("Load Pretrain Model:", pth)
            state_dict = torch.load(pth)
            self.net.load_state_dict(state_dict)

    def save_optim(self, pth, optimizer, epoch, *args, **kwargs):
        pass

    def configure_optimizers(self, *args, **kwargs):
        config_train = self.config['training']
        optimizer = optim.Adam(params=self.net.parameters(), lr=config_train['lr'], betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])

        optimizer_patch1 = optim.Adam(params=self.net_patch1.parameters(), lr=config_train['lr'],
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler_patch1 = optim.lr_scheduler.StepLR(optimizer_patch1, config_train['decay_step'], gamma=config_train['decay_gamma'])

        optimizer_patch2 = optim.Adam(params=self.net_patch2.parameters(), lr=config_train['lr'],
                                      betas=(0.9, 0.999), eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler_patch2 = optim.lr_scheduler.StepLR(optimizer_patch2, config_train['decay_step'],
                                                     gamma=config_train['decay_gamma'])

        return {'optimizer': [optimizer, optimizer_patch1, optimizer_patch2], 'scheduler': [scheduler, scheduler_patch1, scheduler_patch2]}


def train(logger, config):
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    tester = Tester(logger, config, default_oneshot_id=114, collect_sim=False, upsample="bilinear")
    monitor = Monitor(key='mre', mode='dec')
    learner = Learner(logger=logger, config=config)

    patch_sizes = learner.patch_sizes
    id_oneshot = 124
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list = testset.ref_landmarks(id_oneshot)
    dataset_train = Cephalometric(config['dataset']['pth'], patch_sizes=patch_sizes,
                                  pre_crop=False, ref_landmark=landmark_list, use_prob=True, retfunc=2)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    trainer.fit(learner, dataset_train)


def test(logger, config):
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")

    learner = Learner(logger=logger, config=config)
    learner.load(tfilename(config['base']['runs_dir'], 'ckpt', 'best_model_epoch_400.pth'))
    learner.cuda()
    learner.eval()

    # ids = [1,2,3,4,5,6,7,8,9]
    ids = [114, 124, 125, ]
    for id_oneshot in ids:
        res = tester.test(learner, oneshot_id=id_oneshot)
        logger.info(res)
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ssl2/multi_encoder.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print_dict(config)
    # import ipdb; ipdb.set_trace()

    eval(args.func)(logger, config)
