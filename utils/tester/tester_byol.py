import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from PIL import Image, ImageDraw, ImageFont
from datasets.ceph.ceph_test import Test_Cephalometric
from datasets.eval.eval import Evaluater
from utils.utils import visualize

from tutils import tfilename


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
    def __init__(self, logger, config, split="Test1+2", args=None):
        self.config = config
        self.args = args
        self.split = split
        dataset = Test_Cephalometric(config['dataset']['pth'], mode=split)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=1,
                                       shuffle=False, num_workers=config['training']['num_workers'])

        # Creat evluater to record results
        self.evaluater = Evaluater(None, dataset.size, \
                                   dataset.original_size)
        self.id_landmarks = [i for i in range(config['special']['num_landmarks'])]

    def test(self, net, *args, **kwargs):
        # net = net.net.module.net
        return self.test_func(net, *args, **kwargs)

    def test_func(self, net, epoch=None, rank='cuda', oneshot_id=114, dump_label=False, draw=False):
        if net is not None:
            self.model = net
        assert (hasattr(self, 'model'))
        net.eval()
        config = self.config

        one_shot_loader = Test_Cephalometric(pathDataset=self.config['dataset']['pth'], \
                                             mode='Train')
        print(f'ID Oneshot : {oneshot_id}')
        self.evaluater.reset()
        data = one_shot_loader.__getitem__(oneshot_id)
        image, landmarks, im_name = data['img'], data['landmark_list'], data['name']
        feature_list = list()
        image = image.to(rank)
        features_tmp = net(image.unsqueeze(0))
        # import ipdb; ipdb.set_trace()

        # Depth
        feature_list = dict()
        for id_depth in range(6):
            tmp = list()
            for id_mark, landmark in enumerate(landmarks):
                tmpl_y, tmpl_x = landmark[1] // (2 ** (5 - id_depth)), landmark[0] // (2 ** (5 - id_depth))
                mark_feature = features_tmp[id_depth][0, :, tmpl_y, tmpl_x]
                tmp.append(mark_feature.detach().squeeze().cpu().numpy())
            tmp = np.stack(tmp)
            one_shot_feature = torch.tensor(tmp).to(rank)
            feature_list[id_depth] = one_shot_feature

        ID = 1

        for data in self.dataloader:
            img = data['img'].to(rank)
            landmark_list = data['landmark_list']
            features = net(img)

            pred_landmarks_y, pred_landmarks_x = list(), list()
            for id_mark in range(one_shot_feature.shape[0]):
                cos_lists = []
                cos_ori_lists = []
                final_cos = torch.ones_like(img[0, 0]).to(rank)
                for id_depth in range(5):
                    cos_similarity = match_cos(features[id_depth].squeeze(), \
                                               feature_list[id_depth][id_mark])
                    # import ipdb;ipdb.set_trace()
                    cos_similarity = torch.nn.functional.upsample( \
                        cos_similarity.unsqueeze(0).unsqueeze(0), \
                        scale_factor=2 ** (5 - id_depth), mode='nearest').squeeze()
                    # import ipdb;ipdb.set_trace()
                    final_cos = final_cos * cos_similarity
                    cos_lists.append(cos_similarity)
                    # import ipdb;ipdb.set_trace()
                final_cos = (final_cos - final_cos.min()) / \
                            (final_cos.max() - final_cos.min())
                cos_lists.append(final_cos)
                chosen_landmark = final_cos.argmax().item()
                pred_landmarks_y.append(chosen_landmark // 384)
                pred_landmarks_x.append(chosen_landmark % 384)
                debug = torch.cat(cos_lists, 1).cpu()

                a_landmark = landmark_list[id_mark]
                pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)
                if draw:
                    gray_to_PIL2(debug.cpu().detach(), pred_landmark, a_landmark).save( \
                        tfilename(config['base']['runs_dir'], 'visuals', str(ID), f'{id_mark + 1}_debug_w_gt.jpg'))

            preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)]
            self.evaluater.record_old(preds, landmark_list)

            # Optional Save viusal results
            if draw:
                image_pred = visualize(img, preds, landmark_list, num=19)
                image_pred.save(tfilename(config['base']['runs_dir'], 'visuals', str(ID), 'pred.png'))

            if dump_label:
                inference_marks = {id: [int(preds[1][id]), \
                                        int(preds[0][id])] for id in range(19)}
                dir_pth = tfilename(config['base']['runs_dir'], 'pseudo-labels_init')
                with open('{0}/{1:03d}.json'.format(dir_pth, ID), 'w') as f:
                    json.dump(inference_marks, f)
                print("Dumped JSON file:", '{0}/{1:03d}.json'.format(dir_pth, ID))

            ID += 1

        mre = self.evaluater.cal_metrics_all()
        mre['split'] = self.split
        mre['oneshot_id'] = oneshot_id
        return mre
