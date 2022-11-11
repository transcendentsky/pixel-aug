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
from datasets.hand.hand_basic import HandXray
from datasets.eval.eval_hand import Evaluater
from utils.utils import visualize
from tutils import tfilename, trans_args, trans_init, dump_yaml, tdir, timer
from datetime import datetime
from einops import rearrange
from .test_specific_ids import test_specific_ids

torch.multiprocessing.set_sharing_strategy('file_system')


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
    def __init__(self, logger=None, config=None, args=None, mode=None):
        dataset_1 = HandXray(config['dataset']['pth'], mode='Test')
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=0)
        self.oneshot_loader = HandXray(config['dataset']['pth'], mode='Train')
        self.config = config
        self.args = args
        # Creat evluater to record results
        self.evaluater = Evaluater(logger, dataset_1.size)
        self.logger = logger
        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]
        self.dataset_len = len(self.oneshot_loader)

    def log(self, msg, *args, **kwargs):
        if self.logger is None:
            print(msg, *args, **kwargs)
        else:
            self.logger.info(msg, *args, **kwargs)

    def test(self, net, oneshot_ids, draw=True, dump_label=False):
        if net is not None:
            self.model = net
        assert (hasattr(self, 'model'))
        net.eval()
        config = self.config

        if True:
            self.log(f'ID Oneshot : {oneshot_ids}')
            self.evaluater.reset()
            feature_list_list = []
            for oneshot_id in oneshot_ids:
                data = self.oneshot_loader.__getitem__(oneshot_id)
                image = data['img']
                landmarks_ori = data['landmark_list']
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
            for i, data in enumerate(self.dataloader_1):
                img = data['img'].cuda()
                landmark_list = data['landmark_list']
                img_shape = data['img_shape']
                index = data['index']
                print(f"Running --> ", i, end="\r")
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

                    final_cos_multi = torch.stack(final_cos_to_cat, axis=0)
                    max_sim_value = final_cos_multi.max().item()
                    chosen_landmark = final_cos_multi.argmax().item()
                    chosen_landmark = chosen_landmark % (147456)
                    pred_landmarks_x.append(chosen_landmark // 384)
                    pred_landmarks_y.append(chosen_landmark % 384)
                    record_list.append([chosen_landmark // 384, chosen_landmark % 384, max_sim_value])

                record_list_list.append(record_list)
                preds = [np.array(pred_landmarks_x), np.array(pred_landmarks_y)]
                self.evaluater.record_hand(preds, landmark_list, img_shape)

                # Optional Save viusal results
                if draw:
                    image_pred = visualize(img, preds, landmark_list)
                    image_pred.save(tfilename(config['base']['runs_dir'], f'visuals', str(ID), 'pred.png'))

                if dump_label:
                    name = self.dataset.return_name(index[0])
                    inference_marks = {id: [int(preds[1][id]), \
                                            int(preds[0][id])] for id in range(19)}
                    pth = tfilename(config['base']['runs_dir'] + f'/pseudo-labels_init/', f"{name[:-4]}.json")
                    with open(pth, 'w') as f:
                        json.dump(inference_marks, f)
                    print("Dumped JSON file:", pth, end="\r")

                ID += 1

            mre = self.evaluater.cal_metrics()
            return mre, record_list_list


class SimpleEvaler(object):
    def __init__(self, logger, config, all_records, ids_len=3):
        self.config = config
        self.all_records = all_records
        self.ids_len = ids_len

    def debug(self, ids):
        records_x = rearrange(self.all_records[ids, :, :, 0], "n h w -> n (h w)")
        records_y = rearrange(self.all_records[ids, :, :, 1], "n h w -> n (h w)")
        for i, data in enumerate(self.dataloader_1):
            img = data['img']
            landmark_list = data['landmark_list']
            index = data['index']
            img_shape = data['img_shape']

            pred_landmarks_x = records_x[i, :][0]
            pred_landmarks_y = records_y[i, :][0]
            print(records_x)

            preds = [np.array(pred_landmarks_x), np.array(pred_landmarks_y)]
            self.evaluater.record_hand(preds, landmark_list, img_shape)
        mre = self.evaluater.cal_metrics()
        return mre

    def test_multi(self, ids, valid_percent=1.0, dump_label=False):
        # For arbitary ids
        assert len(ids) == self.ids_len
        n, h, w, c = self.all_records.shape  # 150, 150, 19
        assert self.all_records.shape == (609, 300, 37, 3)
        records_x = rearrange(self.all_records[ids, :, :, 0], "n h w -> n (h w)")
        records_y = rearrange(self.all_records[ids, :, :, 1], "n h w -> n (h w)")
        confs = rearrange(self.all_records[ids, :, :, 2], "n h w -> n (h w)")

        conf_idx = np.argsort(confs, axis=0)
        max_conf = np.take_along_axis(confs, conf_idx, axis=0)[-1]
        res_x = np.take_along_axis(records_x, conf_idx, axis=0)[-1]
        res_y = np.take_along_axis(records_y, conf_idx, axis=0)[-1]
        res_x = rearrange(res_x, "(h w) -> h w", h=h, w=w)
        res_y = rearrange(res_y, "(h w) -> h w", h=h, w=w)
        res = np.stack([res_x, res_y], axis=-1)
        # if valid_percent < 1.0:
        #     valid_num = int(150 * valid_percent)
        #     max_conf = np.sort(max_conf)[-valid_num:]

        max_mean_conf = np.mean(max_conf)
        x = np.arange(max_conf.shape[0])

        # Init Dataset
        dataset_1 = HandXray(config['dataset']['pth'], mode='Test')
        dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=0)
        evaluater = Evaluater(logger, dataset_1.size)
        evaluater.reset()

        for i, data in enumerate(dataloader_1):
            img = data['img']
            landmark_list = data['landmark_list']
            index = data['index']
            img_shape = data['img_shape']
            preds = res[i, :, :].transpose((1,0))
            # preds = res[i, :, :]
            evaluater.record_hand(preds, landmark_list, img_shape)

            if dump_label:
                name = dataset_1.return_name(index[0])
                # import ipdb; ipdb.set_trace()
                inference_marks = {id: [int(preds[1][id]), \
                                        int(preds[0][id])] for id in range(19)}
                pth = tfilename(self.config['base']['runs_dir'], f'pseudo_labels_init/', f"{name[:-4]}.json")
                with open(pth, 'w') as f:
                    json.dump(inference_marks, f)

        mre = evaluater.cal_metrics()
        return mre, max_mean_conf, max_conf


def preprocessing(logger, config, args):

    # Tester
    tester = Tester(logger, config)
    net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=config['special']['emb_len'])
    logger.info(f"debug train2.py non local={config['special']['non_local']}")

    ckpt = '/home1/quanquan/code/landmark/code/runs/baseline/ssl_hand/debug/ckpt/best_model_epoch_950.pth'
    config['ckpt'] = ckpt
    assert os.path.exists(ckpt)
    logger.info(f'Load CKPT {ckpt}')
    ckpt1 = torch.load(ckpt)
    net.load_state_dict(ckpt1)
    config['num_ref'] = 1
    net.cuda()

    dump_yaml(logger, config)

    mre_list = []
    ids_list = []
    record_list = []
    timer1 = timer("wtf")
    for i in range(tester.dataset_len):
        ids = [i]
        t = timer1()
        mre, records = tester.test(net, oneshot_ids=ids, dump_label=args.dump, draw=False)
        t = t - timer1()
        print("Using t:", t)
        # records: [150, 19, 3]
        mre_list.append(mre)
        record_list.append(records)
    np.save(tfilename(config['base']['runs_dir'], f"mre_list.npy"), mre_list)
    np.save(tfilename(config['base']['runs_dir'], f"ids_list.npy"), ids_list)
    np.save(tfilename(config['base']['runs_dir'], f"pred_record_list.npy"), record_list)


def review_multi(logger, config, args):
    """ Randomly Choose samples """
    # if not args.review:
    #     preprocessing(logger, config, args)
    num_ref = config['num_reference'] = args.ref

    # pred record is the all predictions of each template, and its confidence, not SIFT Features
    pred_record_pth = tfilename(config['base']['runs_dir'], "pred_record_list.npy")
    pred_record_list = np.load(pred_record_pth)
    print(pred_record_list.shape)

    # eval = SimpleEvaler(logger, config, pred_record_list, 1)
    # debug
    # for i in range(100):
    #     mre = eval.debug([i])
    # import ipdb; ipdb.set_trace()

    if not args.review:
        eval = SimpleEvaler(logger, config, pred_record_list, num_ref)
        mre_list = []
        conf_list = []
        maxconf_list = []
        ids_list = []
        count = 0
        best_res = None
        best_mre = {"mre": 99, "sim":0.0}
        best_sim = {"mre": 99, "sim":0.0}
        valid_percent = config['valid_percent'] = 1.0
        # radius_res_list = [[], [], [], []] # [[]] * 4 bug!
        while True:
            if num_ref > 1:
                ids = np.random.choice(609, size=num_ref, replace=False)
                ids = np.sort(ids)
            else:
                ids = [count]
                if count >= 609: break
            if count >= args.iter:
                break
            count += 1
            assert len(ids) == num_ref
            res, conf, maxconf = eval.test_multi(ids)

            mre = res['mre']
            if mre < best_mre['mre']:
                best_mre = {'mre': mre, "sim": conf, "ids": ids}
                best_res = res
            if conf > best_sim['sim']:
                best_sim = {'mre': mre, "sim": conf, "ids": ids}

            mre_list.append(res['mre'])
            conf_list.append(conf)
            maxconf_list.append(maxconf)
            ids_list.append(ids)

            if count % 20 == 0:
                # print(mre, conf)
                logger.info(f"Best mre dict: {best_mre}, {best_res}")
                logger.info(f"Best sim dict: {best_sim}")
            else:
                print(f"Mre: {res['mre']}, Sim: {conf}")

        np.save(tfilename(config['base']['runs_dir'], f"mre_n{num_ref}_list.npy"), mre_list)
        np.save(tfilename(config['base']['runs_dir'], f"conf_n{num_ref}_list.npy"), conf_list)
        np.save(tfilename(config['base']['runs_dir'], f"ids_n{num_ref}_list.npy"), ids_list)
        np.save(tfilename(config['base']['runs_dir'], f"maxconf_n{num_ref}_list.npy"), maxconf_list)
        # np.save(tfilename(config['base']['runs_dir'], f"radius_res_n{num_ref}_list.npy"), radius_res_list)

    ###################################################################################
    tag = config['base']['tag']
    max_list_pth = config['max_list_pth'] = tfilename(config['base']['runs_dir'], f"conf_n{num_ref}_list.npy")
    res_list = np.load(max_list_pth)

    mre_list_pth = config['mre_list_pth'] = tfilename(config['base']['runs_dir'], f"mre_n{num_ref}_list.npy")
    mre_list = np.load(mre_list_pth)

    draw_scatter(res_list, mre_list, config['base']['runs_dir'] + f"/sort_sort_n{num_ref}_{tag}.pdf", xlabel="sim",
                 ylabel="mre")
    draw_scatter(res_list, mre_list, f"sort_sort_n{num_ref}_{tag}.pdf", xlabel="sim",
                 ylabel="mre")
    print("Saved :", config['base']['runs_dir'] + f"sort_sort_{tag}.png")
    dump_yaml(logger, config)

    res_sort = np.argsort(res_list)[::-1]
    mre_sort = np.argsort(mre_list)
    print("Max sim: ", mre_list[mre_sort[:3]], res_list[mre_sort[:3]])
    print("Max sim: ", mre_list[res_sort[:3]], res_list[res_sort[:3]])


def debug(logger, config, args):
    # ids = [51,164,213,240,338,357,501,506,548,549]
    # ids = [25]
    ids
    pred_record_pth = tfilename(config['base']['runs_dir'], "pred_record_list.npy")
    pred_record_list = np.load(pred_record_pth)
    num_ref = len(ids)
    eval = SimpleEvaler(logger, config, pred_record_list, num_ref)
    res, conf, maxconf = eval.test_multi(ids)
    print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="test_by_multi_hand")
    parser.add_argument("--config", default="configs/ssl/ssl_hand.yaml")
    parser.add_argument("--ref", type=int, default=3)
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--review", action='store_true')
    parser.add_argument("--dump", action='store_true')
    parser.add_argument("--func", default="review_multi")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # if not args.review:
    #     preprocessing(logger, config, args)
    # review_multi(logger, config)
    eval(args.func)(logger, config, args)