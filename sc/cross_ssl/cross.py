"""
    Inter-Image CosSim
"""
import torch
from tutils import trans_args, trans_init, save_script, tfilename
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule, LearnerWrapper
import argparse
from torch import optim
from utils.tester_ssl import Tester
from datasets.ceph_with_sift import Cephalometric
from models.network_emb_study import UNet_Pretrained
from torch import nn
from einops import rearrange



class DesProjector(nn.Module):
    def __init__(self, in_channel=16, out_channel=128):
        super(DesProjector, self).__init__()
        self.projector = nn.Sequential(nn.Linear(in_channel, 256),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(256, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, out_channel)
                                       )

    def forward(self, x):
        return self.projector(x)



cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, nearby=None):
    b, w, h = cos_map.shape
    total_loss = list()
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
        total_loss.append(id_loss)
    return torch.stack(total_loss).mean()


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss


def _tmp_fea_mix(fea_list):
    depth = 0
    # import ipdb; ipdb.set_trace()
    sim_map_0 = torch.nn.functional.upsample( \
        fea_list[depth], \
        scale_factor=2 ** (5 - depth - 1), mode='nearest').squeeze()
    depth = 1
    sim_map_1 = torch.nn.functional.upsample( \
        fea_list[depth], \
        scale_factor=2 ** (5 - depth - 1), mode='nearest').squeeze()
    depth = 2
    sim_map_2 = torch.nn.functional.upsample( \
        fea_list[depth], \
        scale_factor=2 ** (5 - depth - 1), mode='nearest').squeeze()
    depth = 3
    sim_map_3 = torch.nn.functional.upsample( \
        fea_list[depth], \
        scale_factor=2 ** (5 - depth - 1), mode='nearest').squeeze()
    depth = 4
    sim_map_4 = torch.nn.functional.upsample( \
        fea_list[depth], \
        scale_factor=2 ** (5 - depth - 1), mode='nearest').squeeze()
    sim = torch.cat([sim_map_0, sim_map_1, sim_map_2, sim_map_3, sim_map_4], axis=1)
    assert sim.shape[-1] == 192
    return sim


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config, emb_len=16)
        self.net_patch = UNet_Pretrained(3, emb_len=16)
        self.sift_predictor = DesProjector(in_channel=16*5, out_channel=128)
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss(reduction='none')

    def forward(self, x, **kwargs):
        # self.net(x['img'])
        # raise NotImplementedError
        return self.net(x)

    def _tmp_exloss_func(self, raw_fea_list, landmarks, dess, responses, device='cuda'):
        raw_fea_map = _tmp_fea_mix(raw_fea_list)
        # crop_sim_map = _tmp_fea_mix(crop_fea_list)
        # raw_fea_map = rearrange(raw_fea_map, "b c h w -> c b h w")
        # bs = raw_fea_map.shape[0]
        # fea_list = []

        inds = torch.where(landmarks == 1)
        # import ipdb; ipdb.set_trace()
        raw_sims = raw_fea_map[inds[0], :, inds[1], inds[2]]
        response = responses[inds[0], inds[1], inds[2]]
        des = dess[inds[0], :, inds[1], inds[2]]  # shape: n, c
        # import ipdb; ipdb.set_trace()
        pred = self.sift_predictor(raw_sims)
        ex_mse_loss = self.mse_fn(pred, des)
        ex_mse_loss = ex_mse_loss.mean(axis=-1) * response
        loss = ex_mse_loss.mean()
        # import ipdb; ipdb.set_trace()
        return loss

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        crop_imgs = data['crop_imgs']
        raw_loc  = data['raw_loc']
        chosen_loc = data['chosen_loc']
        sift_response = data['sift_response']
        sift_landmark = data['sift_landmark']
        sift_descript = data['sift_descript']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list = self.net_patch(crop_imgs)

        # loss base
        nearby = self.config['training'].get('nearby', None)
        loss_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=nearby)
        loss_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=nearby)
        loss_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=nearby)
        loss_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=nearby)
        loss_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=nearby)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

        # Extra loss
        sift_landmark = torch.round(sift_landmark).long()
        exloss = self._tmp_exloss_func(raw_fea_list, sift_landmark, sift_descript, sift_response) * config['special']['lambda_ex']
        loss = loss + exloss

        return {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1,
                'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4, 'exloss': exloss}

    def load(self):
        state_dict = '/home1/quanquan/code/landmark/code/runs/ssl/ssl/baseline1/ckpt/best_model_epoch_300.pth'
        state_dict = torch.load(state_dict)
        self.net.load_state_dict(state_dict)

    def save_optim(self, pth, optimizer, epoch, *args, **kwargs):
        pass

    def configure_optimizers(self, *args, **kwargs):
        config_train = self.config['training']
        optimizer = optim.Adam(params=self.net.parameters(), lr=config_train['lr'], betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=config_train['weight_decay'])
        optimizer_patch = optim.Adam(params=self.net_patch.parameters(), lr=config_train['lr'],
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])
        scheduler_patch = optim.lr_scheduler.StepLR(optimizer_patch, config_train['decay_step'], gamma=config_train['decay_gamma'])
        return {'optimizer': [optimizer, optimizer_patch], 'scheduler': [scheduler, scheduler_patch]}


def train(logger, config):
    tester = Tester(logger, config)
    monitor = Monitor(key='mre', mode='dec')
    dataset_train = Cephalometric(config['dataset']['pth'])
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/cross_ssl/cross_ssl.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args)
    save_script(config['base']['runs_dir'], __file__)
    eval(args.func)(logger, config)
