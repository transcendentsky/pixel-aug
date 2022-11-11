import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model, CSVLogger
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerWrapper
import argparse
from torch import optim
from utils.tester.tester_hand_ssl import Tester
from datasets.hand.hand_ssl import HandXray
from models.network_emb_study import UNet_Pretrained
import numpy as np


EX_CONFIG = {
    "special": {
        # "pretrain_model": "/home1/quanquan/code/landmark/code/runs/ssl/ssl_hand/run1/ckpt_v/model_latest.pth",
        "patch_size": 192,
        "prob_ratio": 0,
        "entr_t": 0.5,
        "cj_brightness": [0.7,1.5],
        "cj_contrast": [0.7,1.3],
    },
    "training": {
        "load_pretrain_model": False,
        'val_check_interval': 10,
        'num_epochs' : 450,  # epochs
    }
}


torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


# def ce_loss(cos_map, gt_x, gt_y, nearby=None):
#     b, w, h = cos_map.shape
#     total_loss = list()
#     for id in range(b):
#         cos_map[id] = cos_map[id].exp()
#         gt_value = cos_map[id, gt_x[id], gt_y[id]].clone()
#         if nearby is not None:
#             min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
#             min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
#             chosen_patch = cos_map[id, min_x:max_x, min_y:max_y]
#         else:
#             chosen_patch = cos_map[id]
#         id_loss = - torch.log(gt_value / chosen_patch.sum())
#         total_loss.append(id_loss)
#     return torch.stack(total_loss).mean()

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


class Learner(LearnerWrapper):
    def __init__(self, logger, config, *args, **kwargs):
        super(LearnerWrapper, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=config['special']['emb_len'])
        self.net_patch = UNet_Pretrained(3, emb_len=config['special']['emb_len'])
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.net.train()
        self.net_patch.train()

    def eval(self):
        self.net.eval()
        self.net_patch.eval()

    def cuda(self):
        self.net.cuda()
        self.net_patch.cuda()

    def to(self, rank):
        self.net.to(rank)
        self.net_patch.to(rank)

    def forward(self, x, **kwargs):
        # self.net(x['img'])
        # raise NotImplementedError
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        crop_imgs = data['crop_imgs']
        raw_loc  = data['raw_loc']
        chosen_loc = data['chosen_loc']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list = self.net_patch(crop_imgs)

        nearby = self.config['special']['nearby']
        loss_0, gt_value_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
        loss_1, gt_value_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_2, gt_value_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_3, gt_value_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_4, gt_value_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
        loss_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3,
                     'loss_4': loss_4}
        gtvs_dict = {'gtv_0': gt_value_0, 'gtv_1': gt_value_1, 'gtv_2': gt_value_2, 'gtv_3': gt_value_3,
                     'gtv_4': gt_value_4}
        res_dict = {**loss_dict, **gtvs_dict}

        return res_dict

    def load(self, path=None, **kwargs):
        # state_dict = '/home1/quanquan/code/landmark/code/runs/ssl/ssl/debug2/ckpt/best_model_epoch_1820.pth'
        # state_dict = '/home1/quanquan/code/landmark/code/runs/baseline/ssl_hand/debug/ckpt/best_model_epoch_950.pth'
        # state_dict = self.config['base']['runs_dir'] + "ckpt_v/model_latest.pth"
        if path is None:
            path = self.config['special']['pretrain_model']
        state_dict = torch.load(path)
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
    tester = Tester(logger, config, split="Test", upsample="nearest")
    monitor = Monitor(key='mre', mode='dec')
    dataset_train = HandXray(
            config['dataset']['pth'],
            patch_size=config['special']['patch_size'],
            mode="Train",
            cj_brightness=config['special']['cj_brightness'],
            cj_contrast=config['special']['cj_contrast'],
            num_repeat=5)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


def test(logger, config):
    tester = Tester(logger, config, split="Train", upsample="nearest")
    learner = Learner(logger=logger, config=config)
    learner.load(config['base']['runs_dir'] + "ckpt_v/model_latest.pth")
    learner.cuda()
    learner.eval()
    csvlogger = CSVLogger(config['base']['runs_dir'])
    res_dict = tester.test(learner, oneshot_id=21, draw=False)
    logger.info(f"Results: {res_dict}")
    import ipdb; ipdb.set_trace()

    # exit(0)
    mre_list = []
    for i in range(0, 606):
        learner.eval()
        res_dict = tester.test(learner, epoch=i, rank="cuda", oneshot_id=i, draw=i==0)
        # logger.info(f"")
        res_dict['oneshot_id'] = i
        logger.info(f"Results: {res_dict}")
        csvlogger.record(res_dict)
        mre_list.append(res_dict['mre'])
        print("mean: ", np.array(mre_list).mean(), np.median(np.array(mre_list)))
        np.save("./cache/mre_hand.npy", np.array(mre_list))
        # import ipdb; ipdb.set_trace()
        # break


def _get_rank(mre_list):
    mre_list = np.array(mre_list).copy()
    mean = mre_list.mean()
    rank = np.argsort(mre_list)
    total_len = len(mre_list)
    return mean, rank

def test_multi(logger, config, *args, **kwargs):
    tester = Tester(logger, config)
    learner = Learner(logger=logger, config=config)
    learner.load()
    learner.cuda()
    csvlogger = CSVLogger(config['base']['runs_dir'])
    # res_dict = tester.test(learner, oneshot_id=21, draw=True)
    # logger.info(f"Results: {res_dict}")
    # exit(0)
    # dd = {}
    for i in range(0, 200):
        ids = np.random.choice(np.arange(1, 150), config['special']['idnum'], replace=False)
        # dd['oneshot_id'] = ids.tolist()
        # import ipdb; ipdb.set_trace()
        res_dict, record_list = tester.test_multi(learner, epoch=i, rank="cuda", oneshot_ids=ids, draw=i==0)
        # logger.info(f"")
        res_dict['oneshot_id'] = ids.tolist()
        logger.info(f"Results: {res_dict}")
        csvlogger.record(res_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ssl/ssl_hand.yaml")
    parser.add_argument('--idnum', type=int, default=1)
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)

    eval(args.func)(logger, config)
