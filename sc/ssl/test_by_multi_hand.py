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

from networks.network_emb_study import UNet_Pretrained, Wrapper, Probmap_np
from utils_v1.data_loader import Test_Cephalometric
from utils_v2.eval import Evaluater
from utils_v2.utils import visualize
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


class Tester(object):
    def __init__(self, logger, config, net=None, tag=None, train="", args=None):
        mode = "Test1" if train == "" else "Train"

        dataset_1 = Test_Cephalometric(config['dataset_pth'], 'Train')
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=config['num_workers'])
        # # For anthor Testset, deprecated
        # dataset_2 = Cephalometric(config['dataset_pth'], 'Test2')
        # self.dataloader_2 = DataLoader(dataset_2, batch_size=1,
        #                         shuffle=False, num_workers=config['num_workers'])

        self.config = config
        self.args = args

        self.model = net

        # Creat evluater to record results
        self.evaluater = Evaluater(logger, dataset_1.size, \
                                   dataset_1.original_size)

        self.logger = logger

        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['num_landmarks'])]

    def test(self, net, oneshot_ids, draw=True, dump_label=False):
        if net is not None:
            self.model = net
        assert (hasattr(self, 'model'))
        net.eval()
        # net_patch.eval()
        CONF = self.config['conf']
        config = self.config

        if True:
            self.logger.info(f'ID Oneshot : {oneshot_ids}')
            self.evaluater.reset()
            feature_list_list = []
            for oneshot_id in oneshot_ids:
                image, landmarks_ori = self.dataset.__getitem__(oneshot_id)
                image = image.cuda()
                features_tmp = net(image.unsqueeze(0))

                # Depth
                feature_list = dict()
                for id_depth in range(6):
                    tmp = list()
                    for id_mark, landmark in enumerate(landmarks_ori):
                        tmpl_y, tmpl_x = landmark[1] // (2 ** (5 - id_depth)), landmark[0] // (2 ** (5 - id_depth))
                        # print(id_depth, tmpl_x, tmpl_y, landmark, features_tmp[id_depth].shape)
                        # import ipdb; ipdb.set_trace()
                        mark_feature = features_tmp[id_depth][0, :, tmpl_y, tmpl_x]
                        tmp.append(mark_feature.detach().squeeze().cpu().numpy())
                    tmp = np.stack(tmp)
                    one_shot_feature = torch.tensor(tmp).cuda()
                    feature_list[id_depth] = one_shot_feature

                feature_list_list.append(feature_list)
                del feature_list

            ID = 1
            num_marks = one_shot_feature.shape[0]
            record_list_list = []
            for i, (img, landmark_list) in enumerate(self.dataloader_1):
                print(f"Running --> ", i, end="\r")
                img = img.cuda()
                features = net(img)
                record_list = []
                pred_landmarks_x, pred_landmarks_y = list(), list()
                for id_mark in range(num_marks):
                    final_cos_to_cat = []
                    for i, oneshot_id in enumerate(oneshot_ids):
                        feature_list = feature_list_list[i]
                        cos_lists = []
                        final_cos = torch.ones_like(img[0, 0]).cuda()
                        for id_depth in range(5):
                            cos_similarity = match_cos(features[id_depth].squeeze(), \
                                                       feature_list[id_depth][id_mark])
                            cos_similarity = torch.nn.functional.upsample( \
                                cos_similarity.unsqueeze(0).unsqueeze(0), \
                                scale_factor=2 ** (5 - id_depth), mode='nearest').squeeze()
                            # import ipdb;ipdb.set_trace()
                            final_cos = final_cos * cos_similarity
                            cos_lists.append(cos_similarity)
                        final_cos_to_cat.append(final_cos)
                        cos_lists.append(final_cos)

                    # final_cos = (final_cos - final_cos.min()) / \
                    #             (final_cos.max() - final_cos.min())
                    # cos_lists.append(final_cos)
                    final_cos_multi = torch.stack(final_cos_to_cat, axis=0)
                    max_sim_value = final_cos_multi.max().item()
                    chosen_landmark = final_cos_multi.argmax().item()
                    chosen_landmark = chosen_landmark % (147456)
                    pred_landmarks_x.append(chosen_landmark // 384)
                    pred_landmarks_y.append(chosen_landmark % 384)
                    record_list.append([chosen_landmark // 384, chosen_landmark % 384, max_sim_value])
                    if not os.path.isdir(f'visuals/{ID}'):
                        os.makedirs(f'visuals/{ID}')
                    # if draw:
                    #     debug = torch.cat(cos_lists, 1).cpu()
                    #     a_landmark = landmark_list[id_mark]
                    #     pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)
                    #     gray_to_PIL2(debug.cpu().detach(), pred_landmark, a_landmark).save( \
                    #         tfilename('visuals', str(ID), f'{id_mark + 1}_debug_w_gt.jpg'))

                record_list_list.append(record_list)
                preds = [np.array(pred_landmarks_x), np.array(pred_landmarks_y)]
                self.evaluater.record(preds, landmark_list)

                # Optional Save viusal results
                if draw:
                    image_pred = visualize(img, preds, landmark_list)
                    image_pred.save(tfilename(config['runs_dir'], f'visuals', str(ID), 'pred.png'))

                if dump_label:
                    inference_marks = {id: [int(preds[1][id]), \
                                            int(preds[0][id])] for id in range(19)}
                    dir_pth = config['runs_dir'] + f'/pseudo-labels_init/'
                    if not os.path.isdir(dir_pth): os.makedirs(dir_pth)
                    with open('{0}/{1:03d}.json'.format(dir_pth, ID), 'w') as f:
                        json.dump(inference_marks, f)
                    print("Dumped JSON file:", '{0}/{1:03d}.json'.format(dir_pth, ID))

                ID += 1

            mre = self.evaluater.cal_metrics()
            return mre, record_list_list


class SimpleEvaler(object):
    def __init__(self, logger, config, all_records, ids_len=3):
        self.all_records = all_records
        self.ids_len = ids_len
        dataset_1 = Test_Cephalometric(config['dataset_pth'], 'Train')
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=config['num_workers'])
        self.evaluater = Evaluater(logger, dataset_1.size, \
                                   dataset_1.original_size)

    def test(self, ids):
        """ ONLY For two ids """
        # preds for each image
        # for i in range(150):
        #     if i in ids:
        #         break
        records = self.all_records[ids, :, :, :2]
        confs = self.all_records[ids, :, :, 2:]
        res = np.where(confs[0, :, :]>confs[1, :, :], records[0, :, :, :],records[1, :, :, :])
        max_conf = np.where(confs[0, :, :]>confs[1, :, :], confs[0, :, :],confs[1, :, :])
        max_mean_conf = np.mean(max_conf)
        # print(max_conf)
        # import ipdb; ipdb.set_trace()
        self.evaluater.reset()
        for i, (img, landmark_list) in enumerate(self.dataloader_1):
            preds = res[i, :, :].transpose((1,0))
            self.evaluater.record(preds, landmark_list)
        mre = self.evaluater.cal_metrics()

        # for i in ids:
        return mre, max_mean_conf

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
        for i, (img, landmark_list) in enumerate(self.dataloader_1):
            preds = res[i, :, :].transpose((1,0))
            self.evaluater.record(preds, landmark_list)
        mre, rdict = self.evaluater.cal_metrics(ex=True)
        return mre, max_mean_conf, rdict


def test_specific_ids(ids=None):
    from tutils import CSVLogger
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=str, default="")
    args = trans_args(parser)
    logger, config = trans_init(args)
    indices = str(args.indices).split(',')
    indices = [int(ind) for ind in indices]
    csvlogger = CSVLogger(config['base']['runs_dir'], mode="a+")

    # num_ref = 15
    # ids = [12, 32, 35, 36, 45, 78, 95, 96, 122, 128, 132, 133, 136, 139, 140]
    # ids = [20,42,59,91,13]
    ids = indices
    num_ref = len(ids)
    print("len ids: ", num_ref)
    assert len(ids) == num_ref
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--review", action='store_true')
    # args = trans_args(parser)
    # logger, config = trans_init(args)

    # pred record is the all predictions of each template, and its confidence, not SIFT Features
    pred_record_pth = config['pred_record_pth'] = "/home1/quanquan/code/landmark/code/runs-ana/multi_pre1/pred_record_list.npy"
    pred_record_list = np.load(pred_record_pth)
    print(pred_record_list.shape)

    eval = SimpleEvaler(logger, config, pred_record_list, num_ref)
    mre, conf, rdict = eval.test_multi(ids, valid_percent=1.0)
    logger.info(f"Mre: {mre}; conf: {conf}; ")

    mre_list = f"/home1/quanquan/code/landmark/code/runs-ana/multi_sift_analysis/multi_n{num_ref}_1/mre_n{num_ref}_list.npy"
    mre_list = np.load(mre_list)
    tlist = np.concatenate([mre_list, [mre]])
    rank = np.argsort(tlist)
    r = np.where(rank == (len(rank)-1))[0][0]
    # rank_of_rank = np.argsort(rank)
    logger.info(f"Rank of the result: {r}")
    csvlogger.record({
        "indices": indices,
        "mre": mre ,
        "conf": conf,
        "rank": r,
    })

if __name__ == '__main__':
    test_specific_ids()