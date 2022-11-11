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

from datasets.ceph.ceph_test import Test_Cephalometric
from datasets.eval.eval import Evaluater
from utils.utils import visualize
from tutils import tfilename, trans_args, trans_init, dump_yaml, tdir



def match_cos(feature, template):
    feature = feature.permute(1, 2, 0)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    return torch.clamp(cos_similarity, 0, 1)


class Tester(object):
    def __init__(self, logger, config, net=None, tag=None, split="", args=None):
        self.split = split
        dataset_1 = Test_Cephalometric(config['dataset']['pth'], mode=split)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=config['training']['num_workers'])
        self.oneshot_dataset = Test_Cephalometric(config['dataset']['pth'], mode='Train')
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
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]

    def test(self, net, oneshot_ids, draw=False, dump_label=False):
        if net is not None:
            self.model = net
        assert (hasattr(self, 'model'))
        net.eval()
        # net_patch.eval()
        config = self.config

        # if True:
        self.logger.info(f'ID Oneshot : {oneshot_ids}')
        self.evaluater.reset()
        feature_list_list = []
        for oneshot_id in oneshot_ids:
            data = self.oneshot_dataset.__getitem__(oneshot_id)
            image, landmarks_ori, im_name = data['img'], data['landmark_list'], data['name']
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
            print(f"Running --> ", i, end="\r")
            img = data['img'].cuda()
            landmark_list = data['landmark_list']
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
                # if not os.path.isdir(f'visuals/{ID}'):
                #     os.makedirs(f'visuals/{ID}')
                # if draw:
                #     debug = torch.cat(cos_lists, 1).cpu()
                #     a_landmark = landmark_list[id_mark]
                #     pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)
                #     gray_to_PIL2(debug.cpu().detach(), pred_landmark, a_landmark).save( \
                #         tfilename('visuals', str(ID), f'{id_mark + 1}_debug_w_gt.jpg'))

            record_list_list.append(record_list)
            preds = [np.array(pred_landmarks_x), np.array(pred_landmarks_y)]
            self.evaluater.record_old(preds, landmark_list)

            # Optional Save viusal results
            if draw:
                image_pred = visualize(img, preds, landmark_list)
                image_pred.save(tfilename(config['base']['runs_dir'], f'visuals', str(ID), 'pred.png'))

            if dump_label:
                inference_marks = {id: [int(preds[1][id]), \
                                        int(preds[0][id])] for id in range(19)}
                dir_pth = tdir(config['base']['runs_dir'], 'pseudo_labels_init')
                with open('{0}/{1:03d}.json'.format(dir_pth, ID), 'w') as f:
                    json.dump(inference_marks, f)
                print("Dumped JSON file:", '{0}/{1:03d}.json'.format(dir_pth, ID), end="\r")

            ID += 1

        mre = self.evaluater.cal_metrics_all()
        return mre, record_list_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/select/select.yaml")
    parser.add_argument("--ref", type=int, default=3)
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--review", action='store_true')
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # if not args.review:
    #     preprocessing(logger, config)
    # review_multi(logger, config)