# coding: utf-8
"""
    Test by multiple templates,
    Analysis with `stat_ana`
"""
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import json
from PIL import Image, ImageDraw

from models.network_emb_study import UNet_Pretrained, Wrapper, Probmap_np
from datasets.ceph_test import Test_Cephalometric
from utils.eval import Evaluater
from utils.utils import visualize
from tutils import tfilename, trans_args, trans_init, dump_yaml

from einops import rearrange

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
def draw_scatter(points, points2, fname="ttest.png", c="red", set_font=None, xlabel="x", ylabel="y"):
    plt.ioff()  # Turn off interactive plotting off
    if set_font is not None:
        plt.rc('font', family='Times New Roman')
    fig = plt.figure()
    points = points.flatten()
    points2 = points2.flatten()
    plt.scatter(points, points2, c=c, label="???")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()


def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def gray_to_PIL2(tensor, pred_lm, landmark, row=6, width=384):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(images)
    red = (255, 0, 0)
    green = (0, 255, 0)
    # red = 255
    for i in range(row):
        draw.rectangle((pred_lm[0] + i * width - 2, pred_lm[1] - 2, pred_lm[0] + i * width + 2, pred_lm[1] + 2),
                       fill=green)
        draw.rectangle((landmark[0] + i * width - 2, landmark[1] - 2, landmark[0] + i * width + 2, landmark[1] + 2),
                       fill=red)
    draw.line([tuple(pred_lm), tuple(landmark)], fill='green', width=0)
    # import ipdb; ipdb.set_trace()
    return images


def match_cos(feature, template):
    feature = feature.permute(1, 2, 0)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    return torch.clamp(cos_similarity, 0, 1)


def print_vars(vars: np.ndarray):
    for i, var in enumerate(vars):
        # print(var.shape)
        var = var / np.max(var) * 255
        image = Image.fromarray(var.astype(np.uint8))
        image.save(f"visuals/vars/var_{i}.jpg")
        import ipdb;
        ipdb.set_trace()


class SimpleEvaler(object):
    def __init__(self, logger, config, all_records, ids_len=3):
        self.all_records = all_records
        self.ids_len = ids_len
        dataset_1 = Test_Cephalometric(config['dataset']['pth'], 'Train')
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=2)
        self.evaluater = Evaluater(logger, dataset_1.size, \
                                   dataset_1.original_size)

    def test_multi(self, ids, valid_percent=1.0):
        # For arbitary ids
        assert len(ids) == self.ids_len
        # records = self.all_records[ids, :, :, :2]
        n, h, w, c = self.all_records.shape
        records_x = rearrange(self.all_records[ids, :, :, 0], "n h w -> n (h w)")
        records_y = rearrange(self.all_records[ids, :, :, 1], "n h w -> n (h w)")
        confs = rearrange(self.all_records[ids, :, :, 2], "n h w -> n (h w)")

        conf_idx = np.argsort(confs, axis=0)
        # print(conf_idx.shape)
        # import ipdb; ipdb.set_trace()
        max_conf = np.take_along_axis(confs, conf_idx, axis=0)[-1]
        res_x = np.take_along_axis(records_x, conf_idx, axis=0)[-1]
        res_y = np.take_along_axis(records_y, conf_idx, axis=0)[-1]
        res_x = rearrange(res_x, "(h w) -> h w", h=h, w=w)
        res_y = rearrange(res_y, "(h w) -> h w", h=h, w=w)
        res = np.stack([res_x, res_y], axis=-1)
        if valid_percent < 1.0:
            valid_num = int(150 * valid_percent)
            max_conf = np.sort(max_conf)[-valid_num:]
            # print(max_conf, max_conf.shape)

        max_mean_conf = np.mean(max_conf)
        x = np.arange(max_conf.shape[0])
        # draw_scatter(x, max_conf, fname=f"tmp_visuals/max_conf_scatter_{ids[0]}.png")
        # print(max_conf.shape, "Saved scatter Image!")
        # import ipdb; ipdb.set_trace()

        self.evaluater.reset()
        for i, data in enumerate(self.dataloader_1):
            landmark_list = data['landmark_list']
            preds = res[i, :, :].transpose((1,0))
            self.evaluater.record(preds, landmark_list)
        res = self.evaluater.cal_metrics_all()
        return res, max_mean_conf

CONFIG = {
    'dataset':{
        'pth': '/home1/quanquan/datasets/Cephalometric/'
    }
}

def test_specific_ids(indices, logger=None, config=CONFIG):

    ids = indices
    num_ref = len(ids)
    print("len ids: ", num_ref)
    assert len(ids) == num_ref

    # pred record is the all predictions of each template, and its confidence, not SIFT Features
    pred_record_pth = "/home1/quanquan/code/landmark/code/runs-ana/multi_pre1/pred_record_list.npy"
    pred_record_list = np.load(pred_record_pth)
    # print(pred_record_list.shape)

    eval = SimpleEvaler(logger, config, pred_record_list, num_ref)
    res, conf = eval.test_multi(ids, valid_percent=1.0)
    mre = res['mre']

    mre_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{num_ref}_1/mre_n{num_ref}_list.npy"
    mre_list = np.load(mre_list)
    tlist = np.concatenate([mre_list, [mre]])
    rank = np.argsort(tlist)
    r = np.where(rank == (len(rank)-1))[0][0]
    # rank_of_rank = np.argsort(rank)
    # print(f"Rank of the result: {r}")
    _d = {
        "indices": indices,
        "mre": mre ,
        "conf": conf,
        "rank": r,
    }
    return _d


if __name__ == '__main__':
    from tutils import CSVLogger, print_dict
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=str, default="")
    args = trans_args(parser)
    logger, config = trans_init(args)
    print_dict(config['base'])

    indices = str(args.indices).split(',')
    indices = [int(ind) for ind in indices]

    _d = test_specific_ids(indices, logger, config)
    logger.info(_d)

    csvlogger = CSVLogger(config['base']['runs_dir'], mode="a+")
    csvlogger.record(_d)