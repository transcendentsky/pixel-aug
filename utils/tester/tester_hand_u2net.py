
from datasets.hand.hand_basic import HandXray
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from PIL import Image, ImageDraw, ImageFont
from datasets.eval.eval_hand import Evaluater
from utils.utils import visualize
import os
from tutils import tfilename, tdir


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


class Tester(object):
    def __init__(self, logger, config, split="Test", args=None,
                 default_oneshot_id=21,
                 upsample="nearest",):
        self.config = config
        self.default_oneshot_id = default_oneshot_id
        self.upsample = upsample
        self.split = split
        dataset_1 = HandXray(config['dataset']['pth'], split=split, num_repeat=1)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=config['training']['num_workers'])
        self.args = args
        self.logger = logger

        self.evaluater = Evaluater(None, dataset_1.size)
        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]

    def test(self, *args, **kwargs):
        return self.test_func(*args, **kwargs)

    def test_func(self, net, epoch=None, rank='cuda', oneshot_id=None, dump_label=False, draw=False):
        net.eval()
        config = self.config

        oneshot_id = oneshot_id if oneshot_id is not None else self.default_oneshot_id
        print(f'ID Oneshot : {oneshot_id}')
        self.evaluater.reset()
        feature_list = self.get_template_features(oneshot_id, net)

        ID = 1
        num_landmark = 37

        for i, data in enumerate(self.dataloader_1):
            if rank != 'cuda':
                img = data['img'].to(rank)
            else:
                img = data['img'].cuda()
            features = net(img)
            landmark_list = data['landmark_list']
            # img_shape = data['img_shape']
            index = data['index']

            preds = self.test_img_with_one_tmpl(img, features, feature_list)
            # print(preds, landmark_list)
            self.evaluater.record_hand_old2(preds, landmark_list)

            # Optional Save viusal results
            if draw:
                image_pred = visualize(img, preds, landmark_list, num=37)
                image_pred.save(tfilename(config['base']['runs_dir'], 'visuals', str(ID), 'pred.png'))

            ID += 1

        d = self.evaluater.cal_metrics()
        d['split'] = self.split
        d['upsample'] = self.upsample
        d['oneshot_id'] = oneshot_id
        return d

    def test_multi(self, net, oneshot_ids, draw=True, dump_label=False):
        net.eval()
        # CONF = self.config['conf']
        config = self.config
        one_shot_loader = HandXray(self.config['dataset']['pth'], split="Train", datanum=0, ret_mode="no_process")

        if True:
            self.logger.info(f'ID Oneshot : {oneshot_ids}')
            self.evaluater.reset()
            feature_list_list = []
            for oneshot_id in oneshot_ids:
                data = one_shot_loader.__getitem__(oneshot_id)
                image, landmarks, index = data['img'], data['landmark_list'], data['index']
                image = image.cuda()
                features_tmp = net(image.unsqueeze(0))

                # Depth
                feature_list = dict()
                for id_depth in range(5):
                    tmp = list()
                    for id_mark, landmark in enumerate(landmarks):
                        tmpl_y, tmpl_x = landmark[1] // (2 ** (4 - id_depth)), landmark[0] // (2 ** (4 - id_depth))
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
                features = net(img)
                landmark_list = data['landmark_list']
                img_shape = data['img_shape']
                index = data['index']
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
                                scale_factor=2 ** (4 - id_depth), mode=self.upsample).squeeze()
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
                    pred_landmarks_y.append(chosen_landmark // 384)
                    pred_landmarks_x.append(chosen_landmark % 384)
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
                self.evaluater.record_hand_old(preds, landmark_list)

                # Optional Save viusal results
                if draw:
                    image_pred = visualize(img, preds, landmark_list, num=37)
                    image_pred.save(tfilename(config['base']['runs_dir'], f'visuals', str(ID), 'pred.png'))

                if dump_label:
                    inference_marks = {id: [int(preds[1][id]), \
                                            int(preds[0][id])] for id in range(19)}
                    dir_pth = config['base']['runs_dir'] + f'/pseudo-labels_init/'
                    if not os.path.isdir(dir_pth): os.makedirs(dir_pth)
                    with open('{0}/{1:03d}.json'.format(dir_pth, ID), 'w') as f:
                        json.dump(inference_marks, f)
                    print("Dumped JSON file:", '{0}/{1:03d}.json'.format(dir_pth, ID))

                ID += 1

            mre = self.evaluater.cal_metrics()
            return mre, record_list_list

    def get_template_features(self, oneshot_id, net):
        if isinstance(oneshot_id, int):
            return self.get_one_template_feature(oneshot_id, net)
        elif isinstance(oneshot_id, list):
            fea_list = []
            for i in oneshot_id:
                fea = self.get_one_template_feature(i, net)
                fea_list.append(fea)
            return fea_list
        else:
            raise ValueError

    def get_one_template_feature(self, oneshot_id, net):
        assert isinstance(oneshot_id, int), f"Got {oneshot_id}"
        one_shot_loader = HandXray(self.config['dataset']['pth'], split="Train", datanum=0, ret_mode="no_process")
        data = one_shot_loader.__getitem__(oneshot_id)
        image, landmarks, index = data['img'], data['landmark_list'], data['index']
        image = image.cuda()
        features_tmp = net(image.unsqueeze(0))

        print(landmarks)
        # Depth
        feature_list = dict()
        for id_depth in range(5):
            tmp = list()
            for id_mark, landmark in enumerate(landmarks):
                tmpl_y, tmpl_x = landmark[1] // (2 ** (4 - id_depth)), landmark[0] // (2 ** (4 - id_depth))
                # print("template ", tmpl_x, tmpl_y, id_depth)
                mark_feature = features_tmp[id_depth][0, :, tmpl_y, tmpl_x]
                tmp.append(mark_feature.detach().squeeze().cpu().numpy())
            tmp = np.stack(tmp)
            one_shot_feature = torch.tensor(tmp).cuda()
            feature_list[id_depth] = one_shot_feature
        return feature_list

    def test_img_with_tmpls(self, img, features, feature_tmplate):
        if isinstance(feature_tmplate, dict):
            preds = self.test_img_with_one_tmpl(img, features, feature_tmplate)
            return preds
        elif isinstance(feature_tmplate, list):
            preds_list = []
            for a_fea_tmp in feature_tmplate:
                preds = self.test_img_with_one_tmpl(img, features, a_fea_tmp)
                preds_list.append()

    def test_img_with_one_tmpl(self, img, features, feature_tmplate):
        num_landmark = 37
        pred_landmarks_y, pred_landmarks_x = list(), list()
        preds = []
        # confs = []
        for id_mark in range(num_landmark):
            cos_lists = []
            final_cos = torch.ones((384,384)).cuda()
            for id_depth in range(5):
                cos_similarity = match_cos(features[id_depth].squeeze(), \
                                           feature_tmplate[id_depth][id_mark])
                cos_similarity = torch.nn.functional.upsample( \
                    cos_similarity.unsqueeze(0).unsqueeze(0), \
                    scale_factor=2 ** (4 - id_depth), mode=self.upsample).squeeze()
                # import ipdb;ipdb.set_trace()
                final_cos = final_cos * cos_similarity
                cos_lists.append(cos_similarity)
            ## TODO: Here should be changed to unravel_index
            chosen_landmark = final_cos.argmax().item()
            pred_landmarks_y.append(chosen_landmark // 384)
            pred_landmarks_x.append(chosen_landmark % 384)
            # preds.append([chosen_landmark // 384, chosen_landmark % 384, conf])

            # if draw:
            #     debug = torch.cat(cos_lists, 1).cpu()
            #     a_landmark = landmark_list[id_mark]
            #     pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)
            #     gray_to_PIL2(debug.cpu().detach(), pred_landmark, a_landmark).save( \
            #         tfilename(config['base']['runs_dir'], 'visuals', str(ID), f'{id_mark + 1}_debug_w_gt.jpg'))

        preds = [np.array(pred_landmarks_x), np.array(pred_landmarks_y)]
        # print("test_img ", chosen_landmark, final_cos.shape)
        # import ipdb; ipdb.set_trace()
        return preds