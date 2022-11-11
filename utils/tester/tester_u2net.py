"""
    Copied from tester_ssl for debug

"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from PIL import Image, ImageDraw, ImageFont
from datasets.ceph.ceph_test import Test_Cephalometric
from datasets.eval.eval import Evaluater
from utils.utils import visualize, visualize_landmarks
from utils.utils import voting

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
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-8)
    return torch.clamp(cos_similarity, 0, 1)


def resolve_max_sim_list(max_list):
    temp = np.array(max_list)
    mean = np.mean(temp)
    # import ipdb; ipdb.set_trace()
    return mean


def resolve_near_sim_list(near_list):
    temp = np.array(near_list)  # 150, 19, 3, 8
    # print(temp.shape)
    # import ipdb; ipdb.set_trace()
    mean = np.mean(temp, axis=0)
    mean = np.mean(mean, axis=0)
    mean = np.mean(mean, axis=-1)
    return mean


def resolve_max_sim_layer(max_list):
    # max_list: 150, 19, 5
    temp = np.array(max_list)
    # import ipdb; ipdb.set_trace()
    mean = np.mean(temp, axis=0)
    mean = np.mean(mean, axis=0)
    return mean


class Tester(object):
    def __init__(self,
                 logger,
                 config,
                 net=None,
                 tag=None,
                 split="Test1+2",
                 args=None,
                 retfunc=1,
                 default_oneshot_id=114,
                 upsample="bilinear",
                 collect_sim=False,
                 size=None,
                 collect_near=False):
        self.split = split
        self.retfunc = retfunc
        self.config = config
        self.args = args
        self.model = net
        self.default_oneshot_id = default_oneshot_id
        self.upsample = upsample
        self.collect_sim = collect_sim
        self.collect_near = collect_near
        print("Warning! using tester_ssl_debug! ")
        dataset_1 = Test_Cephalometric(config['dataset']['pth'], mode=split)
        # print("Test")
        self.dataset = dataset_1
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=config['training']['num_workers'])

        # Creat evluater to record results
        self.evaluater = Evaluater(None, dataset_1.size, dataset_1.original_size)
        # self.evaluater = Evaluater(logger, [384, 384],
        #                            [2400, 1935])
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]

        # Eval on multiple layers
        self.evaluater_layers = {f"layer_{i}": Evaluater(None, dataset_1.size, dataset_1.original_size) for i in range(5)}

    def test(self, *args, **kwargs):
        if self.collect_sim:
            return self.test_func_collect_sim(*args, **kwargs)
        if self.retfunc == 1:
            return self.test_func(*args, **kwargs)
        elif self.retfunc == 2:
            return self.test_func2(*args, **kwargs)
        else:
            raise ValueError

    def test_func_collect_sim(self, net, epoch=None, rank='cuda', oneshot_id=-1, dump_label=False, draw=False):
        res1 = self.test_func(net, epoch, rank, oneshot_id, dump_label, draw, collect_details=True)

        mean_max1 = resolve_max_sim_list(res1['max_sim'])
        mean_lm1 = resolve_max_sim_list(res1['lm_sim'])
        mean_mean = resolve_max_sim_list(res1['mean_sim'])
        mean_max2_1 = resolve_max_sim_layer(res1['max_sim_layer1'])
        mean_max3_1 = resolve_max_sim_layer(res1['max_sim_layer2'])
        mean_mean2 = resolve_max_sim_layer(res1['mean_sim_layer'])
        print(" ----------------------------- ")
        # print(mean_mean.shape, mean_max1.shape)
        # import ipdb; ipdb.set_trace(0)
        # new_dict = {"mre": res1['mre'], "SDR 2": res1["SDR 2"], "SDR 2.5": res1["SDR 2.5"],
        #             "SDR 3": res1["SDR 3"], "SDR 4": res1["SDR 4"],
        #             "oneshot_id": res1['oneshot_id'], "split": res1["split"]}
        res1['mean_sim'] = mean_mean
        res1['max_sim'] = mean_max1
        res1['lm_sim'] = mean_lm1

        if self.collect_near:
            mean_near = resolve_near_sim_list(res1['near_sim'])
            # res1['near_sim'] = mean_near
            for i, v in enumerate(mean_near):
                res1[f'near_sim_{i}'] = v
        for i in range(5):
            res1[f'sim_layer_point_{i}'] = mean_max2_1[i]
            res1[f'sim_layer_max_{i}'] = mean_max3_1[i]
            res1[f'sim_layer_mean_{i}'] = mean_mean2[i]

        res1.pop('max_sim_layer1')
        res1.pop('max_sim_layer2')
        res1.pop('mean_sim_layer')
        res1.pop('near_sim')
        res1.pop('cos')
        return res1

    def test_template_sim(self, *args, **kwargs):
        one_shot_loader = Test_Cephalometric(pathDataset=self.config['dataset']['pth'], mode='oneshot_debug', default_oneshot_id=kwargs['oneshot_id'])
        assert len(one_shot_loader) == 1, f"Got {len(one_shot_loader)}"
        self.dataloader_1 = DataLoader(one_shot_loader, batch_size=1,
                                       shuffle=False, num_workers=self.config['training']['num_workers'])
        return self.test_func_collect_sim(*args, **kwargs)

    def test_func(self, net, epoch=None, rank='cuda', oneshot_id=-1, dump_label=False, draw=False, collect_details=False):
        if net is not None:
            self.model = net
        assert (hasattr(self, 'model'))

        oneshot_id = oneshot_id if oneshot_id >=0 else self.default_oneshot_id
        net.eval()
        config = self.config

        one_shot_loader = Test_Cephalometric(pathDataset=self.config['dataset']['pth'], mode='Train')
        print(f'ID Oneshot : {oneshot_id}')
        self.evaluater.reset()
        data = one_shot_loader.__getitem__(oneshot_id)
        image, landmarks, im_name = data['img'], data['landmark_list'], data['name']
        temp_index_debug = data['name']
        feature_list = list()
        if rank != 'cuda':
            image = image.to(rank)
        else:
            image = image.cuda()
        features_tmp = net(image.unsqueeze(0))

        # Depth
        feature_list = dict()
        for id_depth in range(5):
            tmp = list()
            for id_mark, landmark in enumerate(landmarks):
                tmpl_y, tmpl_x = landmark[1] // (2 ** (4 - id_depth)), landmark[0] // (2 ** (4 - id_depth))
                # print(id_depth, tmpl_y, tmpl_x, features_tmp[id_depth].shape)
                mark_feature = features_tmp[id_depth]
                # print("1")
                mark_feature = mark_feature[0, :, tmpl_y, tmpl_x]
                # print("2")
                tmp.append(mark_feature.detach().squeeze().cpu().numpy())
            tmp = np.stack(tmp)
            if rank != 'cuda':
                one_shot_feature = torch.tensor(tmp).to(rank)
            else:
                one_shot_feature = torch.tensor(tmp).cuda()
            feature_list[id_depth] = one_shot_feature

        # import ipdb; ipdb.set_trace()
        cos_list_list_list = []
        max_list_list = []
        mean_list_list = []
        near_list_list = []
        max_in_layer_list_list = []
        max_per_layer_list_list = []
        mean_in_layer_list_list = []
        lm_list_list = []
        for ID, data in enumerate(self.dataloader_1):
            ID += 1
            data_index = data['name']
            # print("#################  template, data index: ", temp_index_debug, data_index)
            # import ipdb; ipdb.set_trace()
            if rank != 'cuda':
                img = data['img'].to(rank)
            else:
                img = data['img'].cuda()
            features = net(img)
            landmark_list = data['landmark_list']
            pred_lm = [] # standard landmark format
            cos_list_list = []
            max_list = []
            mean_list = []
            near_list = []
            max_in_layer_list = []
            max_per_layer_list = []
            mean_in_layer_list = []
            lm_list = []
            pred_layer = {f"layer_{i}": [] for i in range(5)}
            pred_landmarks_y, pred_landmarks_x = list(), list()
            for id_mark in range(one_shot_feature.shape[0]):
                cos_lists = []
                cos_ori_lists = []
                max_sim_all_layers = []
                if rank != 'cuda':
                    final_cos = torch.ones_like(img[0, 0]).to(rank)
                else:
                    final_cos = torch.ones_like(img[0, 0]).cuda()
                for id_depth in range(5):
                    cos_similarity = match_cos(features[id_depth].squeeze(), \
                                               feature_list[id_depth][id_mark])
                    # import ipdb;ipdb.set_trace()
                    cos_similarity = torch.nn.functional.upsample( \
                        cos_similarity.unsqueeze(0).unsqueeze(0), \
                        scale_factor=2 ** (4 - id_depth), mode=self.upsample).squeeze()
                    # import ipdb;ipdb.set_trace()
                    final_cos = final_cos * cos_similarity
                    cos_lists.append(cos_similarity)
                cos_lists.append(final_cos)

                ## TODO: Here should be changed to unravel_index
                assert tuple(final_cos.shape[-2:]) == (384,384), f"Got {final_cos.shape}"
                chosen_landmark = final_cos.argmax().item()
                pred_landmarks_y.append(chosen_landmark // 384)
                pred_landmarks_x.append(chosen_landmark % 384)
                pred_lm.append([chosen_landmark // 384, chosen_landmark % 384])

                if collect_details:
                    # cos_list_list.append(cos_lists.detach().cpu().numpy())
                    final_cos = final_cos.clamp(0, 1)
                    max_sim = final_cos[chosen_landmark // 384, chosen_landmark % 384]
                    max_list.append(max_sim.detach().cpu().numpy())
                    # for i in range(5):
                    #     print(f"layer {i}: target sim {cos_lists[i][chosen_landmark // 384, chosen_landmark % 384]} , max sim {cos_lists[i].max().item()}")
                    # assert max_sim >= 1.0, f"Got max_sim {max_sim}"
                    mean_list.append(final_cos.mean().detach().cpu().numpy())

                    if self.collect_near:
                        near_sim = []
                        loc_list_0 = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]).astype(int)
                        loc_list = [loc_list_0, loc_list_0*2, loc_list_0 * 3]
                        for a_loc_list in loc_list:
                            near_sim_0 = []
                            for i, a_loc in enumerate(a_loc_list):
                                h, w = chosen_landmark // 384 + a_loc[0], chosen_landmark % 384 + a_loc[1]
                                value = float(final_cos[h, w].detach().cpu().item())
                                # print(value)
                                # import ipdb; ipdb.set_trace()
                                near_sim_0.append(value)
                            assert len(near_sim_0) == 8, f"Got {len(near_sim)}, {a_loc_list}"
                            near_sim.append(near_sim_0)
                        near_list.append(near_sim)
                    # print(near_list)
                    a_list = []
                    b_list = []
                    c_list = []
                    for layer_i, cos in enumerate(cos_lists[:-1]):
                        cos = cos.clamp(0, 1)
                        value_of_pred = cos[chosen_landmark // 384, chosen_landmark % 384].detach().cpu().numpy()
                        max_value = cos.max().detach().cpu().numpy()
                        mean_value = cos.mean().detach().cpu().numpy()
                        a_list.append(value_of_pred)
                        b_list.append(max_value)
                        c_list.append(mean_value)

                        assert len(cos.shape) == 2, f"Got {cos.shape}"
                        pred_i = np.unravel_index(cos.argmax().item(), cos.shape)
                        pred_layer[f'layer_{layer_i}'].append(pred_i)

                    max_in_layer_list.append(a_list)
                    max_per_layer_list.append(b_list)
                    mean_in_layer_list.append(c_list)

                    a_landmark = landmark_list[id_mark]
                    landmark_sim = final_cos[a_landmark[1], a_landmark[0]]
                    lm_list.append(landmark_sim.detach().cpu().numpy())
                    # import ipdb; ipdb.set_trace()

                if draw:
                    debug = torch.cat(cos_lists, 1).cpu()
                    a_landmark = landmark_list[id_mark]
                    pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)
                    gray_to_PIL2(debug.cpu().detach(), pred_landmark, a_landmark).save( \
                        tfilename(config['base']['runs_dir'], 'visuals', str(ID), f'{id_mark + 1}_debug_w_gt.jpg'))

            if collect_details:
                cos_list_list_list.append(cos_list_list)
                max_list_list.append(max_list)
                lm_list_list.append(lm_list)
                mean_list_list.append(mean_list)
                near_list_list.append(near_list)
                max_in_layer_list_list.append(max_in_layer_list)
                max_per_layer_list_list.append(max_per_layer_list)
                mean_in_layer_list_list.append(mean_in_layer_list)

                for i in range(5):
                    self.evaluater_layers[f'layer_{i}'].record_new(pred_layer[f'layer_{i}'], landmark_list)

            preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)]
            self.evaluater.record_old(preds, landmark_list)
            # pred_lm_list.append(pred_lm)

            # Optional Save viusal results
            if draw:
                # image_pred = visualize(img, preds, landmark_list)
                import ipdb; ipdb.set_trace()
                image_pred = visualize_landmarks(img=img, preds=pred_lm, gts=landmark_list, num=19, draw_line=True)
                image_pred.save(tfilename(config['base']['runs_dir'], 'visuals', str(ID), 'pred.png'))

            if dump_label:
                inference_marks = {id: [int(preds[1][id]), \
                                        int(preds[0][id])] for id in range(19)}
                dir_pth = tdir(config['base']['runs_dir'], 'pseudo_labels_init')
                with open('{0}/{1:03d}.json'.format(dir_pth, ID), 'w') as f:
                    json.dump(inference_marks, f)
                print("Dumped JSON file:", '{0}/{1:03d}.json'.format(dir_pth, ID), end="\r")

        mre = self.evaluater.cal_metrics_all()
        if collect_details:
            mre_all = self.evaluater.cal_metrics_per_lm()
            # import ipdb; ipdb.set_trace()
            for i in range(5):
                res = self.evaluater_layers[f'layer_{i}'].cal_metrics_all()
                for k, v in res.items():
                    mre[f'l{i}_{k}'] = v

        if collect_details:
            return {**mre, **mre_all, "oneshot_id": oneshot_id, "split": self.split,
                "cos":cos_list_list_list, "max_sim":max_list_list, "lm_sim":lm_list_list,
                "max_sim_layer1": max_in_layer_list_list, "max_sim_layer2": max_per_layer_list_list,
                "mean_sim": mean_list_list, "mean_sim_layer": mean_in_layer_list_list,
                "near_sim": near_list_list, "upsample": self.upsample}
        else:
            return {**mre, "oneshot_id": oneshot_id, "split": self.split, "upsample": self.upsample}
