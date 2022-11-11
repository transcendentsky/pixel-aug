"""
    debug the point probmap / and others
"""

import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl_adapm import Cephalometric
from models.network_emb_study import UNet_Pretrained


torch.backends.cudnn.benchmark = True

EX_CONFIG = {
    "dataset":{
        'entr': '/home1/quanquan/datasets/Cephalometric/entr1/train/',
        'prob': '/home1/quanquan/datasets/Cephalometric/prob_pseudo/train/',
        'edge': '/home1/quanquan/datasets/Cephalometric/edge1/train/',
    },
    "special":{
        "patch_size": 64,
        "prob_ratio": 0,
        "entr_t": 0.3,
        'probmap_ks': 192,
        'num_repeat': 10,
        'inverse': True,
    },
    "training": {
        'save_mode': ['best', 'latest'],
        'val_check_interval': 50,
        'num_epochs': 600,
    }
}


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
        self.net_patch = UNet_Pretrained(3, emb_len=16)
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()
        # self.set_hook()

    def set_hook(self):
        for item in self.net.named_parameters():
            if item[0] == 'trans_5.weight':
                print("Register ", item[0])
                h = item[1].register_hook(lambda grad: print("Grad value: ", item[0], torch.einsum('ijkl->i', grad).item(), grad.shape))
            else:
                # print("Ignore ", item[0])
                pass

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

        # import ipdb; ipdb.set_trace()
        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

        loss_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3,
                     'loss_4': loss_4}
        gtvs_dict = {'gtv_0': gt_value_0, 'gtv_1': gt_value_1, 'gtv_2': gt_value_2, 'gtv_3': gt_value_3,
                     'gtv_4': gt_value_4}
        res_dict = {**loss_dict, **gtvs_dict}

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
        optimizer_patch = optim.Adam(params=self.net_patch.parameters(), lr=config_train['lr'],
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])
        scheduler_patch = optim.lr_scheduler.StepLR(optimizer_patch, config_train['decay_step'], gamma=config_train['decay_gamma'])
        return {'optimizer': [optimizer, optimizer_patch], 'scheduler': [scheduler, scheduler_patch]}


def train(logger, config):
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    # ------------------------
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id, collect_sim=False, split="Test1+2", upsample="bilinear")
    monitor = Monitor(key='mre', mode='dec')

    # testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    # all_landmarks = []
    # for i in range(len(testset)):
    #     landmark_list = testset.ref_landmarks(i)
    #     all_landmarks.append(landmark_list)

    testset = Test_Cephalometric("/home1/quanquan/code/landmark/code/runs/ssl/dump_label_from_ssl/ssl_baseline/pseudo_labels_init", mode="pseudo", pre_crop=False)
    all_landmarks = []
    for i in range(len(testset)):
        landmark_list = testset.ref_pseudo_landmarks(i)
        all_landmarks.append(landmark_list)

    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['special']['patch_size'],
                             entr_map_dir=config['dataset']['entr'], prob_map_dir=config['dataset']['prob'],
                             mode="Train", retfunc=2, hard_select=True, multi_output=False, num_repeat=config['special']['num_repeat'],
                                  runs_dir=config['base']['runs_dir'])
    # dataset_train.prob_map_point(all_landmarks)
    # dataset_train.prob_map_for_all(all_landmarks, sharpness=0.2, kernel_size=config['special']['probmap_ks'], inverse=config['special']['inverse'])
    # dataset_train.entr_map_from_image(temperature=config['special']['entr_t'], inverse=True)
    # dataset_train.entr_map_ushape(temperature=config['special']['entr_t'], inverse=config['special']['inverse'])
    dataset_train.entr_map_ushape2(temperature=config['special']['entr_t'], inverse=config['special']['inverse'])
    # dataset_train.edge_map_from_image(config['dataset']['edge'], inverse=True)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


def test(logger, config):
    template_oneshot_id = 114
    tester = Tester(logger, config, default_oneshot_id=template_oneshot_id, collect_sim=False, split="Train", upsample="bilinear")
    learner = Learner(logger, config)
    learner.load("/home1/quanquan/code/landmark/code/runs/ssl/ssl_probmap7/prob_ks1/ckpt_v/model_best.pth")
    learner.cuda()
    learner.eval()
    res = tester.test(learner)
    logger.info(f"{res}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print_dict(config)
    # import ipdb; ipdb.set_trace()

    eval(args.func)(logger, config)
