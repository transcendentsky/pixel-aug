import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from PIL import Image
import numpy as np

# from models.network_emb_study import UNet_Pretrained
from models.network_cs import UNet_SSL, UNet_Pretrained
from datasets.ceph.ceph_ssl import Cephalometric
from utils.tester.tester_ssl_cs import Tester
from tutils import trans_args, trans_init, save_image, dump_yaml, load_yaml, tfilename, print_dict, CSVLogger, MetricLogger, save_script
from einops import rearrange
from torchvision.utils import save_image


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


def L1Loss(pred, gt, mask=None):
    # L1 Loss for offset map
    assert (pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
    return distence.sum() / mask.sum()


def gray_to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def focal_loss(pred, gt):
    eps = 1e-8
    return (-(1 - pred) * gt * torch.log(pred + eps) - pred * (1 - gt) * torch.log(1 - pred + eps)).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--config", default='configs/ssl/ssl_cs.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    save_script(config['base']['runs_dir'], __file__, logger=logger)
    dump_yaml(logger, config)
    # print_dict(config)
    runs_dir = config['base']['runs_dir']
    # train_indices = str(config['base']['indices']).split(',')
    # train_indices = [int(ind) for ind in train_indices]

    net = UNet_SSL(3, non_local=config['special']['non_local'], emb_len=16)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    net_patch = UNet_Pretrained(3, emb_len=16)
    net_patch = torch.nn.DataParallel(net_patch)
    net_patch = net_patch.cuda()

    start_epoch = 0
    config_train = config['training']
    optimizer = optim.Adam(params=net.parameters(), \
                           lr=config_train['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])

    optimizer2 = optim.Adam(params=net_patch.parameters(), \
                            lr=config_train['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler2 = StepLR(optimizer2, config_train['decay_step'], gamma=config_train['decay_gamma'])

    loss_logic_fn = focal_loss
    loss_regression_fn = L1Loss

    # Tester
    tester = Tester(logger, config)

    dataset = Cephalometric(config['dataset']['pth'], mode="Train", retfunc=3)
    # sampler = torch.utils.data.sampler.SubsetRandomSampler(np.array(train_indices))
    dataloader = DataLoader(dataset, batch_size=config_train['batch_size'],
                            num_workers=config_train['num_workers'], drop_last=True)  # , sampler=sampler
    csvlogger = CSVLogger(tfilename(config['base']['runs_dir'], 'csv'), mode='a+')
    metriclogger = MetricLogger(delimiter=" ", logger=logger)

    # for data in metriclogger.log_every(range(20), print_freq=1, header="[*]"):
    #     i = 0
    #     metriclogger.update(**{"ind": i, "data": data})
    # results = metriclogger.return_final_dict()

    for epoch in range(start_epoch, config_train['num_epochs']):
        net.train()
        net_patch.train()
        iter_index = 0
        print_freq = 10 if epoch == 0 else 300
        for data in metriclogger.log_every(dataloader, print_freq=print_freq, header=""):
            raw_imgs = data['raw_imgs'].cuda()
            crop_imgs = data['crop_imgs'].cuda()
            raw_loc = data['raw_loc']
            chosen_loc = data['chosen_loc']
            mask = data['mask'].cuda()
            offset_y = data['offset_y'].cuda()
            offset_x = data['offset_x'].cuda()
            # import ipdb; ipdb.set_trace()
            # save_image(mask[0], f"tmp/mask.png")
            # save_image(offset_y[0], f"tmp/offset_y.png")
            # save_image(offset_x[0], f"tmp/offset_x.png")

            crop_fea_list = net_patch(crop_imgs, ret_last_layer=True)
            tmpl_feature = torch.stack([crop_fea_list[-1][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                        for id in range(raw_loc.shape[0])]).squeeze()
            assert len(tmpl_feature.shape) == 2, f"Got tmpl_feature.shape {tmpl_feature.shape}"
            raw_fea_list = net(raw_imgs, tmpl_feature)

            nearby = config['special']['nearby']
            loss_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
            loss_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_ssl = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

            heatmap = raw_fea_list[6]
            regression_y = raw_fea_list[7]
            regression_x = raw_fea_list[8]
            indices = np.array([1, 4, 9, 16, 25, 36, 49, 64]) - 1
            heatmap = rearrange(heatmap, "b c h w -> (b c) h w")[indices]
            regression_y = rearrange(regression_y, "b c h w -> (b c) h w")[indices]
            regression_x = rearrange(regression_x, "b c h w -> (b c) h w")[indices]

            # import ipdb; ipdb.set_trace()
            logic_loss = loss_logic_fn(heatmap, mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)
            loss_prompt = regression_loss_x + regression_loss_y + logic_loss * 2 # config_train['lambda']

            # loss = loss_ssl + loss_prompt * 0.004
            loss = logic_loss
            # loss = regression_loss_y + regression_loss_x

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()

            # import ipdb; ipdb.set_trace()
            # iter_index += 1
            log_dict = {
                "loss": loss,
                "loss_ssl": loss_ssl,
                "loss_prompt": loss_prompt,
                "loss_0": loss_0,
                "loss_1": loss_1,
                "loss_2": loss_2,
                "loss_3": loss_3,
                "loss_4": loss_4,
                "logic_loss": logic_loss,
                "regloss_x": regression_loss_x,
                "regloss_y": regression_loss_y,
            }
            metriclogger.update(**log_dict)
            if epoch == 0:
                print("Pipeline Debug!")
                break

        results = metriclogger.return_final_dict()
        logger.info(f"Epoch {epoch}: {results} ")

        scheduler.step()
        scheduler2.step()

        if (epoch + 0) % config_train['save_interval'] == 0 and epoch >= 0:
            print("Testing! ")
            _d = tester.test(net, net_patch, epoch=epoch)
            _d['record_time'] = datetime.now()
            _d['epoch'] = epoch
            # _d['test_mode'] = args.data
            csvlogger.record(_d)
            logger.info(f"Record: {_d}")
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net_patch.state_dict(), runs_dir + "/model_patch_epoch_{}.pth".format(epoch))
            # import ipdb; ipdb.set_trace()
    # # Test
    _d = tester.test(net)
    _d['record_time'] = datetime.now()
    _d['epoch'] = epoch
    _d['test_mode'] = "Test1+2"
    csvlogger.record(_d)
