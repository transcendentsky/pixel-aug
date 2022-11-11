"""
    Train with both Ceph and Hand datasets
"""

import argparse
import os
from pathlib import Path
import yaml
import yamlloader

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss

from models.network_emb_study import UNet_Pretrained, Probmap, Wrapper, Probmap_np
# from datasets.ceph_ssl import Cephalometric, Test_Cephalometric
from datasets.mixed.ceph_hand import CephAndHand_SSL
# from mylogger import get_mylogger, set_logger_dir
from utils.tester.tester_ssl import Tester
from utils.tester.tester_ceph_hand_ssl import Tester_mixed
from PIL import Image
import numpy as np

from tutils import trans_init, trans_args, dump_yaml, timer, tenum, count_model, CSVLogger
from tutils.trainer import Recorder
from torch.cuda.amp import autocast, GradScaler


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

def cos_visual(tensor):
    tensor = torch.clamp(tensor, 0, 10)
    tensor = tensor * 25.5
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def ce_loss(cos_map, gt_y, gt_x, nearby=None):
    b, h, w = cos_map.shape
    total_loss = list()
    for id in range(b):
        cos_map[id] = cos_map[id].exp()
        gt_value = cos_map[id, gt_y[id], gt_x[id]].clone()
        if nearby is not None:
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            chosen_patch = cos_map[id, min_y:max_y, min_x:max_x]
        else:
            chosen_patch = cos_map[id]
        id_loss = - torch.log(gt_value / chosen_patch.sum())
        total_loss.append(id_loss)
    # print(torch.stack(total_loss).mean())
    return torch.stack(total_loss).mean()


def match_inner_product1(feature, template):
    feature = feature.permute(0, 2, 3, 1)
    template = template.unsqueeze(1).unsqueeze(1)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    # print(cos_similarity.max(), cos_similarity.min())
    cos_similarity = torch.clamp(cos_similarity, 0., 1.)
    assert torch.max(cos_similarity) <= 1.0, f"Maximum Error, Got max={torch.max(cos_similarity)}"
    assert torch.min(cos_similarity) >= 0.0, f"Maximum Error, Got max={torch.min(cos_similarity)}"
    return cos_similarity


cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)


def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim

def dump_best_config(logger, config, info):
    # dump yaml
    config = {**config, **info}
    with open(config['base']['runs_dir'] + "/best_config.yaml", "w") as f:
        yaml.dump(config, f)
    logger.info("Dump best config")


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    # ii = 0
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss


def dump_label_from_ssl(logger, config, model):
    tester = Tester(logger, config)
    tester.test(model, dump_label=True)

# CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ablation_nearby --pretrain --tag v2_ablation_near7 --nearby 7
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--config", default="configs/ssl/ceph_hand.yaml", help="default configs")
    # args = parser.parse_args()
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    logger.info(config)

    config_base = config['base']
    config_train = config['training']
    config_special = config['special']

    tag = config_base['tag']
    runs_dir = config_base['runs_dir']
    id_oneshot = config_train['oneshot_id'] = 126

    # Tester
    tester = Tester(logger, config, tag=args.tag)

    dataset = CephAndHand_SSL(img_dir1=config['dataset']['pth'], img_dir2=config['dataset2']['pth'], patch_size=config_special['patch_size'])
    dataloader = DataLoader(dataset, batch_size=config_train['batch_size'],
                            drop_last=True, shuffle=True, num_workers=config_train['num_workers'])

    # import ipdb; ipdb.set_trace()
    net = UNet_Pretrained(3, non_local=config_special['non_local'], emb_len=config_special['emb_len'])
    logger.info(f"debug train2.py non local={config_special['non_local']}")
    net_patch = UNet_Pretrained(3, emb_len=config_special['emb_len'])

    net = net.cuda()
    net_patch = net_patch.cuda()

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=config_train['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])
    optimizer_patch = optim.Adam(params=net_patch.parameters(), \
                                 lr=config_train['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler_patch = StepLR(optimizer_patch, config_train['decay_step'], gamma=config_train['decay_gamma'])

    # loss
    loss_logic_fn = torch.nn.CrossEntropyLoss()
    mse_fn = torch.nn.MSELoss()

    # Best MRE record
    best_mre = 100.0
    best_epoch = -1
    dump_yaml(logger, config)
    nearby = config_special['nearby']

    batch_recorder = Recorder(reduction="sum")
    timer_epoch = timer("epoch")
    timer_batch = timer("in batch")
    csvlogger = CSVLogger(runs_dir)

    # Use AMP accelaration
    scalar = GradScaler()

    for epoch in range(config_train['num_epochs']):
        net.train()
        net_patch.train()
        logic_loss_list = list()
        batch_recorder.clear()
        count = 0
        for time_load, index, data in tenum(dataloader):
            count += 1
            raw_imgs = data['raw_imgs']
            crop_imgs = data['crop_imgs']
            raw_loc  = data['raw_loc']
            chosen_loc = data['chosen_loc']

            timer_batch()
            raw_imgs = raw_imgs.cuda()
            crop_imgs = crop_imgs.cuda()
            time_cuda = timer_batch()

            with autocast():
                raw_fea_list = net(raw_imgs)
                crop_fea_list = net_patch(crop_imgs)
                time_forward = timer_batch()

                loss_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
                loss_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
                loss_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
                loss_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
                loss_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
                loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
                time_loss = timer_batch()

                optimizer.zero_grad()
                optimizer_patch.zero_grad()
                scalar.scale(loss).backward()
                scalar.step(optimizer)
                scalar.step(optimizer_patch)
                scalar.update()

                time_bp = timer_batch()

                logic_loss_list.append(np.array([loss_0.cpu().item(), loss_1.cpu().item(), \
                                                 loss_2.cpu().item(), loss_3.cpu().item(), loss_4.cpu().item()]))
                time_record = timer_batch()
                batch_recorder.record({'time_load': time_load, "time_cuda":time_cuda,
                                       "time_fd": time_forward, "time_loss":time_loss,
                                       "time_bp":time_bp, "time_record": time_record, })
            # if epoch == 0:
            #     print("Check code")
            #     break
        _dict = batch_recorder.cal_metrics()
        print("Count : ", count)

        time_epoch = timer_epoch()
        losses = np.stack(logic_loss_list).transpose()
        # import ipdb; ipdb.set_trace()
        logger.info("Epoch {} Training logic loss {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} t:{:.3f}". \
                    format(epoch, losses[0].mean(), losses[1].mean(), losses[2].mean(), \
                           losses[3].mean(), losses[4].mean(), time_epoch))
        print(_dict)

        scheduler.step()
        scheduler_patch.step()

        if (epoch) % config_train['save_interval'] == 3:
            net.eval()
            net_patch.eval()
            _dict = tester.test(net, epoch=epoch, dump_label=False, oneshot_id=id_oneshot)
            mre = _dict['mre']
            if mre < best_mre:
                best_mre = mre
                best_epoch = epoch
                logger.info("Achieve New Record! ")
                save_dict = {"epoch": best_epoch, "best_mre": best_mre,
                             "model": runs_dir + "/model_epoch_{}.pth".format(epoch),
                             "model_patch": runs_dir + "/model_patch_epoch_{}.pth".format(epoch),
                             }
                save_dict = {**_dict, **save_dict}
                # dump_best_config(logger, config, save_dict)
                csvlogger.record(save_dict)
                torch.save(net.state_dict(), runs_dir + "/model_best_epoch_{}.pth".format(epoch))
                torch.save(net_patch.state_dict(), runs_dir + "/model_best_patch_epoch_{}.pth".format(epoch))

            logger.info(
                f"tag:{tag} ***********  Best MRE:{best_mre} in Epoch {best_epoch} || Epoch:{epoch}:{mre} ***********")
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_latest")
            torch.save(net_patch.state_dict(), runs_dir + "/model_patch_latest")
            # net.probmap.save()

            # config_train['last_epoch'] = epoch
        # dump yaml
        # with open(runs_dir + "/config.yaml", "w") as f:
        #     yaml.dump(config, f)
