import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from PIL import Image, ImageDraw, ImageFont
from datasets.ceph.ceph_test import Test_Cephalometric
from datasets.eval.eval import Evaluater
from utils.utils import visualize
from utils.utils import voting
from einops import rearrange

from tutils import tfilename, tdir


def output_ensemble_by_voting(confs, records_x, records_y, **kwargs):
    # conf_shape: N, C
    conf_idx = np.argsort(confs, axis=0)
    max_conf = np.take_along_axis(confs, conf_idx, axis=0)[-1]
    res_x = np.take_along_axis(records_x, conf_idx, axis=0)[-1]
    res_y = np.take_along_axis(records_y, conf_idx, axis=0)[-1]
    return res_x, res_y


def test_output_ensemble(ids, all_records):
    n, h, w, c = all_records.shape  # 150, 150, 19
    records_x = rearrange(all_records[ids, :, :, 0], "n h w -> n (h w)")
    records_y = rearrange(all_records[ids, :, :, 1], "n h w -> n (h w)")
    confs = rearrange(all_records[ids, :, :, 2], "n h w -> n (h w)")

    conf_idx = np.argsort(confs, axis=0)
    # print(conf_idx.shape)
    # import ipdb; ipdb.set_trace()
    max_conf = np.take_along_axis(confs, conf_idx, axis=0)[-1]
    res_x = np.take_along_axis(records_x, conf_idx, axis=0)[-1]
    res_y = np.take_along_axis(records_y, conf_idx, axis=0)[-1]
    res_x = rearrange(res_x, "(h w) -> h w", h=h, w=w)
    res_y = rearrange(res_y, "(h w) -> h w", h=h, w=w)
    res = np.stack([res_x, res_y], axis=-1)


def gray_to_PIL2(tensor, pred_lm, landmark, row=6, width=384):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = img.fromarray(tensor.int().numpy().astype(np.uint8)).convert('RGB')
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
    def __init__(self,
                 logger,
                 config,
                 net=None,
                 tag=None,
                 split="Train",
                 train="",
                 args=None,
                 retfunc=1,
                 default_oneshot_id=124):
        self.split = split
        self.config = config
        self.retfunc = retfunc
        dataset_eval = Test_Cephalometric(config['dataset']['pth'], mode=split, preprocess=True)
        self.dataset = dataset_eval
        self.dataloader_1 = DataLoader(dataset_eval, batch_size=1,
                                       shuffle=False, num_workers=config['training']['num_workers'])
        self.one_shot_loader = Test_Cephalometric(pathDataset=config['dataset']['pth'], mode='Train', preprocess=True)
        self.args = args
        self.model = net
        self.default_oneshot_id = default_oneshot_id

        # Creat evluater to record results
        self.evaluater = Evaluater(None, dataset_eval.size, \
                                   dataset_eval.original_size)

        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]

    def test(self, *args, **kwargs):
        if self.retfunc == 1:
            return self.test_func(*args, **kwargs)
        elif self.retfunc == 2:
            return self.test_func2(*args, **kwargs)
        else:
            raise ValueError

    def test_func(self, net, epoch=None, rank='cuda', oneshot_id=-1, dump_label=False, draw=False):
        if net is not None:
            self.model = net
        assert (hasattr(self, 'model'))

        oneshot_id = oneshot_id if oneshot_id >=0 else self.default_oneshot_id
        net.eval()
        config = self.config
        num_landmark = 19

        print(f'ID Oneshot : {oneshot_id}')
        self.evaluater.reset()
        data = self.one_shot_loader.__getitem__(oneshot_id)
        img, landmarks, im_name = data['img'], data['landmark_list'], data['name']

        preprocess_names = ['AutoContrast', 'Contrast', 'Equalize', 'Posterize2', ]
        img_ac = data['img_AutoContrast']
        img_co = data['img_Contrast']
        img_eq = data['img_Equalize']
        img_po = data['img_Posterize2']
        extra_img_list = [img_ac, img_co, img_eq, img_po]
        img_total = torch.stack([img] + extra_img_list[:1])
        # print("DEBUG: ", len(extra_img_list[:-2]))
        # import ipdb; ipdb.set_trace()
        img_total = img_total.cuda()
        features_tmp = net(img_total)

        # Depth
        features_list_total = []
        for i in range(img_total.size(0)):
            feature_list = dict()
            for id_depth in range(6):
                tmp = list()
                for id_mark, landmark in enumerate(landmarks):
                    tmpl_y, tmpl_x = landmark[1] // (2 ** (5 - id_depth)), landmark[0] // (2 ** (5 - id_depth))
                    mark_feature = features_tmp[id_depth][i, :, tmpl_y, tmpl_x]
                    tmp.append(mark_feature.detach().squeeze().cpu().numpy())
                tmp = np.stack(tmp)
                one_shot_feature = torch.tensor(tmp).cuda()
                feature_list[id_depth] = one_shot_feature
            features_list_total.append(feature_list)

        # import ipdb; ipdb.set_trace()
        record_list_list = []
        for ID, data in enumerate(self.dataloader_1):
            ID = ID + 1
            img_ac = data['img_AutoContrast']
            img_co = data['img_Contrast']
            img_eq = data['img_Equalize']
            img_po = data['img_Posterize2']
            extra_img_list = [img_ac, img_co, img_eq, img_po]
            img = data['img']
            # import ipdb; ipdb.set_trace()
            img_total = torch.cat([img] + extra_img_list[:1], dim=0)
            img_total = img_total.cuda()
            # assert img_total.size(0) == 5
            features = net(img_total)
            landmark_target = data['landmark_list']
            record_list = []

            pred_landmarks_y, pred_landmarks_x = list(), list()
            for id_mark in range(num_landmark):
                cos_lists = []
                final_cos_to_cat = []
                for img_idx in range(len(features_list_total)):
                    feature_template = features_list_total[img_idx]
                    cos_lists = []
                    final_cos = torch.ones(img.shape[-2:]).cuda()
                    for id_depth in range(5):
                        cos_similarity = match_cos(features[id_depth][img_idx].squeeze(), \
                                                   feature_template[id_depth][id_mark])
                        # print("Debug: ", id_depth, img_idx, cos_similarity.shape)
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
            self.evaluater.record_old(preds, landmark_target)

            # debug = torch.cat(cos_lists, 1).cpu()
            # a_landmark = landmark_target[id_mark]
            # pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)
            # if draw:
            #     gray_to_PIL2(debug.cpu().detach(), pred_landmark, a_landmark).save( \
            #         tfilename(config['base']['runs_dir'], 'visuals', str(ID), f'{id_mark + 1}_debug_w_gt.jpg'))

            # Optional Save viusal results
            if draw:
                image_pred = visualize(img, preds, landmark_target)
                image_pred.save(tfilename(config['base']['runs_dir'], 'visuals', str(ID), 'pred.png'))

            if dump_label:
                inference_marks = {id: [int(preds[1][id]), \
                                        int(preds[0][id])] for id in range(19)}
                dir_pth = tdir(config['base']['runs_dir'], 'pseudo_labels_init')
                with open('{0}/{1:03d}.json'.format(dir_pth, ID), 'w') as f:
                    json.dump(inference_marks, f)
                print("Dumped JSON file:", '{0}/{1:03d}.json'.format(dir_pth, ID), end="\r")


        mre = self.evaluater.cal_metrics_all()
        return {**mre, "oneshot_id": oneshot_id, "split": self.split}

    def test_func2(self, net, rank='cuda', oneshot_id=126, dump_label=False, draw=False):
        net.eval()
        config = self.config

        one_shot_loader = Test_Cephalometric(pathDataset=self.config['dataset']['pth'], \
                                             mode='Train')
        print(f'ID Oneshot : {oneshot_id}')
        self.evaluater.reset()
        data = one_shot_loader.__getitem__(oneshot_id)
        img, landmarks, im_name = data['img'], data['landmark_list'], data['name']
        features_tmp = net(img.unsqueeze(0))

        tmpl_feature = torch.stack([features_tmp[-1][[id], :, landmark[id][0], landmark[id][1]] \
                                    for id, landmark in enumerate(landmarks)]).squeeze()

        for data in self.dataloader_1:
            img = data['img'].cuda()
            landmark_target = data['landmark_list']

            raw_fea_list = net(img, tmpl_feature)
            heatmap, regression_y, regression_x = raw_fea_list[5], raw_fea_list[6], raw_fea_list[7]

            pred_landmark, votings = voting( \
                heatmap, regression_y, regression_x, self.Radius, get_voting=True)
            self.evaluater.record(pred_landmark, landmark_target)

        return self.evaluater.cal_metrics_all()