"""
    For modification to check debugs
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
from datasets.old_data_loader_gp_v2 import Cephalometric, Test_Cephalometric
# from mylogger import get_mylogger, set_logger_dir
from utils.old_test_conf import Tester
from PIL import Image
import numpy as np

from tutils import trans_init, trans_args, dump_yaml, timer, tenum, count_model
from tutils.trainer import Recorder

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
    gt_y, gt_x = raw_y // (2 ** 2), raw_x // (2 ** 2)
    tmpl_y, tmpl_x = chosen_y // (2 ** 2), chosen_x // (2 ** 2)
    tmpl_feature = torch.stack([crop_fea_list[3][[id], :, tmpl_y[id], tmpl_x[id]] \
                                for id in range(gt_y.shape[0])]).squeeze()
    ret_inner_2 = match_inner_product(raw_fea_list[3], tmpl_feature)
    loss_2 = ce_loss(ret_inner_2, gt_y, gt_x, nearby=config_train['nearby'])

def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    ii = 0
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss



# CUDA_VISIBLE_DEVICES=4 python -m scripts.train_ablation_nearby --pretrain --tag v2_ablation_near7 --nearby 7
if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config", default="configs/ssl/old.yaml", help="default configs")
    parser.add_argument("--test", action='store_true', help="Test Mode")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--pretrain", type=str, default='')  # emb-16-289 cconf_ctr 1060
    # parser.add_argument('--nearby', type=int, default=-1)
    parser.add_argument('--emb_len', type=int, default=-1)
    parser.add_argument('--use_prob', action='store_true')
    # args = trans_args(parser)
    args = parser.parse_args()
    logger, config = trans_init(args)
    logger.info(config)

    config_base = config['base']
    config_train = config['training']

    tag = config_base['tag']
    use_prob = args.use_prob
    runs_dir = config_base['runs_dir']
    ret_func = config_train['ret_func'] = 2 if use_prob else 1
    id_oneshot = config_train['oneshot_id'] = 126
    # print(args)
    # import ipdb; ipdb.set_trace()

    # Tester
    tester = Tester(logger, config, tag=args.tag)
    # Dataset
    # if use_prob:
    #     testset = Test_Cephalometric('../../dataset/Cephalometric/', mode="Oneshot", pre_crop=False, id_oneshot=id_oneshot) # 126
    #     landmark_list = testset.ref_landmarks(0)
    # else: landmark_list = None
    landmark = None
    dataset = Cephalometric(config['dataset']['pth'], 'Train', patch_size=config_train['patch_size'], retfunc=ret_func, use_prob=use_prob, ref_landmark=landmark_list)
    dataloader = DataLoader(dataset, batch_size=config_train['batch_size'],
                            drop_last=True, shuffle=True, num_workers=config_train['num_workers'])

    net = UNet_Pretrained(3, non_local=config_train['non_local'], emb_len=config_train['emb_len'])
    # probmap = Probmap_np(config)
    # net = Wrapper(net=unet, probmap=probmap)
    logger.info(f"debug train2.py non local={config_train['non_local']}")
    net_patch = UNet_Pretrained(3, emb_len=config_train['emb_len'])

    # print(net)
    # print(net_patch)
    # example_input = torch.randn(1, 3, 224, 224)
    # print(count_model(net, example_input))
    # print(count_model(net_patch, example_input))
    # import ipdb; ipdb.set_trace()
    # clossfn = CompLoss()

    if False:
        pretrain_config_pth = '/data/quanquan/oneshot/runs2/' + args.pretrain + '/best_config.yaml'
        with open(pretrain_config_pth) as f:
            pretrain_config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
        pretrain_epoch = config['pretrain_epoch'] = pretrain_config['epoch']
        pretrain_tag = config['pretrain_tag'] = pretrain_config['tag']
        ckpt = "/data/quanquan/oneshot/runs2/" + pretrain_tag + f"/model_epoch_{pretrain_epoch}.pth"
        assert os.path.exists(ckpt), f"{ckpt}"
        print(f"Load CKPT: ", ckpt)
        logger.info(f'Load CKPT {ckpt}')
        ckpt = torch.load(ckpt)
        net.load_state_dict(ckpt)
        ckpt2 = "/data/quanquan/oneshot/runs2/" + pretrain_tag + f"/model_patch_epoch_{pretrain_epoch}.pth"
        ckpt2 = torch.load(ckpt2)
        net_patch.load_state_dict(ckpt2)

    # if args.resume:
    #     epoch = -1
    #     ckpt = runs_dir + f"/model_epoch_{epoch}.pth"
    #     assert os.path.exists(ckpt)
    #     logger.info(f'Load CKPT {ckpt}')
    #     ckpt = torch.load(ckpt)
    #     net.load_state_dict(ckpt)
    #     ckpt2 = runs_dir + f"/model_patch_epoch_{epoch}.pth"
    #     assert os.path.exists(ckpt2)
    #     ckpt2 = torch.load(ckpt2)
    #     net_patch.load_state_dict(ckpt2)
    net = net.cuda()
    net_patch = net_patch.cuda()

    if args.test:
        epoch = 109
        ckpt = runs_dir + f"/model_epoch_{epoch}.pth"
        print(f'Load CKPT {ckpt}')
        ckpt = torch.load(ckpt)
        net.load_state_dict(ckpt)
        tester.test(net, epoch=epoch)
        exit()

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=config_train['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])

    optimizer_patch = optim.Adam(params=net_patch.parameters(), \
                                 lr=config_train['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler_patch = StepLR(optimizer_patch, config_train['decay_step'], gamma=config_train['decay_gamma'])

    # loss
    loss_logic_fn = torch.nn.CrossEntropyLoss()
    mse_fn = torch.nn.MSELoss()

    # Best MRE record
    best_mre = 100.0
    best_epoch = -1
    alpha = config_train['alpha'] = 0.99
    b = config_train['batch_size']
    CONF = config_train['conf']
    dump_yaml(logger, config)

    batch_recorder = Recorder(reduction="sum")
    timer_epoch = timer("epoch")
    timer_batch = timer("in batch")


    for epoch in range(config_train['num_epochs']):
        net.train()
        net_patch.train()
        logic_loss_list = list()
        batch_recorder.clear()
        for time_load, index, (raw_img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x) in tenum(dataloader):
            # print("debug", raw_img.shape, crop_imgs.shape, chosen_x.shape)
            # import ipdb; ipdb.set_trace()
            with torch.autograd.set_detect_anomaly(False):
                timer_batch()
                raw_img = raw_img.cuda()
                crop_imgs = crop_imgs.cuda()
                time_cuda = timer_batch()

                raw_fea_list = net(raw_img)
                crop_fea_list = net_patch(crop_imgs)
                time_forward = timer_batch()

                # loss base
                gt_y, gt_x = raw_y // (2 ** 5), raw_x // (2 ** 5)
                tmpl_y, tmpl_x = chosen_y // (2 ** 5), chosen_x // (2 ** 5)

                tmpl_feature = torch.stack([crop_fea_list[0][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_5 = match_inner_product(raw_fea_list[0], tmpl_feature)  # shape [8,12,12]

                loss_5 = ce_loss(ret_inner_5, gt_y, gt_x)

                # loss base
                gt_y, gt_x = raw_y // (2 ** 4), raw_x // (2 ** 4)
                tmpl_y, tmpl_x = chosen_y // (2 ** 4), chosen_x // (2 ** 4)
                tmpl_feature = torch.stack([crop_fea_list[1][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_4 = match_inner_product(raw_fea_list[1], tmpl_feature)
                loss_4 = ce_loss(ret_inner_4, gt_y, gt_x, nearby=config_train['nearby'])

                # loss base
                gt_y, gt_x = raw_y // (2 ** 3), raw_x // (2 ** 3)
                tmpl_y, tmpl_x = chosen_y // (2 ** 3), chosen_x // (2 ** 3)
                tmpl_feature = torch.stack([crop_fea_list[2][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_3 = match_inner_product(raw_fea_list[2], tmpl_feature)
                loss_3 = ce_loss(ret_inner_3, gt_y, gt_x, nearby=config_train['nearby'])

                # loss base
                gt_y, gt_x = raw_y // (2 ** 2), raw_x // (2 ** 2)
                tmpl_y, tmpl_x = chosen_y // (2 ** 2), chosen_x // (2 ** 2)
                tmpl_feature = torch.stack([crop_fea_list[3][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_2 = match_inner_product(raw_fea_list[3], tmpl_feature)
                loss_2 = ce_loss(ret_inner_2, gt_y, gt_x, nearby=config_train['nearby'])

                # loss base
                gt_y, gt_x = raw_y // (2 ** 1), raw_x // (2 ** 1)
                tmpl_y, tmpl_x = chosen_y // (2 ** 1), chosen_x // (2 ** 1)
                tmpl_feature = torch.stack([crop_fea_list[4][[id], :, tmpl_y[id], tmpl_x[id]] \
                                            for id in range(gt_y.shape[0])]).squeeze()
                ret_inner_1 = match_inner_product(raw_fea_list[4], tmpl_feature)
                loss_1 = ce_loss(ret_inner_1, gt_y, gt_x, nearby=config_train['nearby'])

                loss = loss_5 + loss_4 + loss_3 + loss_2 + loss_1
                time_loss = timer_batch()

                optimizer.zero_grad()
                optimizer_patch.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_patch.step()
                time_bp = timer_batch()

                logic_loss_list.append(np.array([loss_5.cpu().item(), loss_4.cpu().item(), \
                                                 loss_3.cpu().item(), loss_2.cpu().item(), loss_1.cpu().item()]))
                time_record = timer_batch()
                batch_recorder.record({'time_load': time_load, "time_cuda":time_cuda,
                                       "time_fd": time_forward, "time_loss":time_loss,
                                       "time_bp":time_bp, "time_record": time_record, })
            # if epoch == 0:
            #     print("Check code")
            #     break
        _dict = batch_recorder.cal_metrics()

        time_epoch = timer_epoch()
        losses = np.stack(logic_loss_list).transpose()
        # import ipdb; ipdb.set_trace()
        logger.info("Epoch {} Training logic loss {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} t:{:.3f}". \
                    format(epoch, losses[0].mean(), losses[1].mean(), losses[2].mean(), \
                           losses[3].mean(), losses[4].mean(), time_epoch))
        print(_dict)

        if CONF:
            logger.info("Mean conf matrix: m5 , {}, {}, {}, {}, {}".format(np.mean(net.probmap.conf_5), \
                                                                           np.mean(net.probmap.conf_4),
                                                                           np.mean(net.probmap.conf_3),
                                                                           np.mean(net.probmap.conf_2),
                                                                           np.mean(net.probmap.conf_1)))
            print(f"conf_5 {net.probmap.conf_5}")

        scheduler.step()
        scheduler_patch.step()

        if (epoch) % config_train['save_seq'] == 3:
            net.eval()
            net_patch.eval()
            _dict = tester.test(net, epoch=epoch, dump_label=False, oneshot_id=id_oneshot)
            mre = _dict['mre']
            if mre < best_mre:
                best_mre = mre
                best_epoch = epoch
                logger.info("Achieve New Record! ")
                save_dict = {"epoch": best_epoch, "mre": best_mre,
                             "model": runs_dir + "/model_epoch_{}.pth".format(epoch),
                             "model_patch": runs_dir + "/model_patch_epoch_{}.pth".format(epoch),
                             }
                dump_best_config(logger, config, save_dict)

            logger.info(
                f"tag:{tag} ***********  Best MRE:{best_mre} in Epoch {best_epoch} || Epoch:{epoch}:{mre} ***********")
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net_patch.state_dict(), runs_dir + "/model_patch_epoch_{}.pth".format(epoch))
            # net.probmap.save()

            config_train['last_epoch'] = epoch
        # dump yaml
        # with open(runs_dir + "/config.yaml", "w") as f:
        #     yaml.dump(config, f)
