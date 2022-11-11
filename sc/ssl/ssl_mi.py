import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model, tenum
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl_mi import Cephalometric
from models.network_emb_study import UNet_Pretrained
import math
import numpy as np
from tutils import torchvision_save, print_dict


torch.backends.cudnn.benchmark = True

cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, nearby=None, gmask=None):
    b, w, h = cos_map.shape
    assert cos_map.shape == gmask.shape, f"shapes: {cos_map.shape}, {gmask.shape}"

    # import ipdb; ipdb.set_trace()
    gt_values_to_record = []
    chosen_patch_list = []
    total_loss = list()
    for id in range(b):
        # assert gmask[id, gt_x[id], gt_y[id]] == 1.0, f"got: {gmask[id, gt_x[id], gt_y[id]]}, max: {gmask[id].max()}"
        cos_map[id] = cos_map[id].exp()
        if nearby is not None:
            assert nearby == 9, f"nearby = {nearby}"
            gt_value = cos_map[id, gt_x[id], gt_y[id]].clone()
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            chosen_patch = cos_map[id, min_x:max_x, min_y:max_y]
            if gmask is not None:
                gmask0 = gmask[id, min_x:max_x, min_y:max_y]
                # gmask0 = gmask0.max() - gmask0
                chosen_patch += gmask0.cuda() * chosen_patch.clone()
        else:
            assert w == 12
            chosen_patch = cos_map[id]
            gt_value = cos_map[id, gt_x[id], gt_y[id]].clone()
            if gmask is not None:
                gmask0 = gmask[id, :, :]
                # gmask0 = gmask0.max() - gmask0
                # chosen_patch += gmask0.cuda() * chosen_patch.clone()

        id_loss = - torch.log(gt_value.mean() / chosen_patch.sum())
        # print("loss", id_loss, " batch:", id, " gt_value:", gt_value.mean())
        total_loss.append(id_loss)
        # chosen_patch_list.append(chosen_patch)
        gt_values_to_record.append(gt_value.clone().detach().log().cpu())
    gt_values_to_record = torch.stack(gt_values_to_record).mean()
    return torch.stack(total_loss).mean(), gt_values_to_record, None #, chosen_patch_list


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None, gmask=None):
    # print("loss level: ", ii)
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss, gt_values = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby, gmask=gmask)
    return loss, gt_values #, chosen_patch_list


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config['training']['non_local'], emb_len=16)
        self.net_patch = UNet_Pretrained(3, emb_len=16)
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()

    def forward(self, x, **kwargs):
        # self.net(x['img'])
        # raise NotImplementedError
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        crop_imgs = data['crop_imgs']
        raw_loc  = data['raw_loc']
        chosen_loc = data['chosen_loc']
        gmask_0 = data['gmask_0']
        gmask_1 = data['gmask_1']
        gmask_2 = data['gmask_2']
        gmask_3 = data['gmask_3']
        gmask_4 = data['gmask_4']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list = self.net_patch(crop_imgs)

        nearby = self.config['special']['nearby']

        loss_0, gtv_0, patch_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, gmask=None)
        loss_1, gtv_1, patch_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, gmask=None)
        loss_2, gtv_2, patch_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, gmask=None)
        loss_3, gtv_3, patch_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, gmask=None)
        loss_4, gtv_4, patch_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, gmask=gmask_0)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
        loss_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4}
        gtvs_dict = {'gtv_0': gtv_0, 'gtv_1': gtv_1, 'gtv_2': gtv_2, 'gtv_3': gtv_3, 'gtv_4': gtv_4}
        res_dict = {**loss_dict, **gtvs_dict}
        return res_dict #, [patch_0, patch_1, patch_2, patch_3, patch_4]

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
        optimizer_patch = optim.Adam(params=self.net_patch.parameters(), lr=config_train['lr'],
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])
        scheduler_patch = optim.lr_scheduler.StepLR(optimizer_patch, config_train['decay_step'], gamma=config_train['decay_gamma'])
        return {'optimizer': [optimizer, optimizer_patch], 'scheduler': [scheduler, scheduler_patch]}


def retain_grad(plist):
    for p in plist:
        for pp in p:
            pp.retain_grad()


def watch_and_change_grad(plist):
    patch_0, patch_1, patch_2, patch_3, patch_4 = plist
    print(patch_1[0].grad)
    # import ipdb; ipdb.set_trace()
    pass


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MyTrainer, self).__init__(*args, **kwargs)

    def train(self, model, trainloader, epoch, optimizer, scheduler=None, do_training_log=True):
        model.train()
        out = {}
        if do_training_log and self.logging_available:
            self.recorder.clear()
            time_record = 0.1111
            self.timer_batch()

        for load_time, batch_idx, data in tenum(trainloader):
            model.on_before_zero_grad()
            optimizer.zero_grad()
            self.timer_data()
            # training steps
            for k, v in data.items():
                if type(v) == torch.Tensor:
                    data[k] = v.to(self.rank)
            time_data_cuda = self.timer_data()
            if self.use_amp:
                raise NotImplemented
                with autocast():
                    self.timer_net()
                    out = model.training_step(data, batch_idx)
                    assert isinstance(out, dict)
                    time_fd = self.timer_net()
                    loss = out['loss']
                    self.scalar.scale(loss).backward()
                    self.scalar.step(optimizer)
                    self.scalar.update()
                    time_bp = self.timer_net()
            else:
                self.timer_net()
                out, plist = model.training_step(data, batch_idx)
                if torch.isnan(out['loss']):
                    print("Nan Value: ", out['loss'])
                    raise ValueError(f"Get loss: {out['loss']}")
                assert isinstance(out, dict)
                time_fd = self.timer_net()
                loss = out['loss']

                if batch_idx == 0:
                    retain_grad(plist)
                loss.backward()
                if batch_idx == 0:
                    watch_and_change_grad(plist)

                import ipdb; ipdb.set_trace()
                optimizer.step()
                time_bp = self.timer_net()

            time_batch = self.timer_batch()
            # batch logger !
            if do_training_log and self.logging_available:
                out['time_load'] = load_time
                out['time_cuda'] = time_data_cuda
                out['time_forward'] = time_fd
                out['time_bp'] = time_bp
                out['time_record'] = time_record
                out['time_batch'] = time_batch
                self.timer_data()
                self.recorder.record(out)
                time_record = self.timer_data()
            # for debug !
            if epoch == 0:
                if self.logging_available:
                    self.logger.info("[*] Debug Checking Pipeline !!!")
                break
            model.on_after_zero_grad(d=out)
        if scheduler is not None:
            scheduler.step()


def train(logger, config):
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    tester = Tester(logger, config, default_oneshot_id=114, collect_sim=True, upsample="bilinear", collect_near=False)
    monitor = Monitor(key='mre', mode='dec')

    id_oneshot = 114
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list = testset.ref_landmarks(id_oneshot)
    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['training']['patch_size'],
                                  pre_crop=False, ref_landmark=landmark_list, use_prob=True, retfunc=2)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
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
    parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print_dict(config['base'])

    eval(args.func)(logger, config)
