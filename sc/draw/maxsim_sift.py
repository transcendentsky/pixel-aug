import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.network_emb_study import UNet_Pretrained, Probmap, Wrapper, Probmap_np
import torch
import os
from datasets.ceph.ceph_test import Test_Cephalometric
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from utils.utils import visualize
from tutils import trans_init, trans_args, dump_yaml, tfilename


def select_descrete_points_by_box(points):
    """ By boxes  """
    grid = (24,24)
    size = 384
    interval = 384 // 24
    boxes = np.zeros(grid)
    selected_points = []
    unselected = []
    for point in points:
        point = point // interval
        if boxes[point[0], point[1]] >= 1:
            boxes[point[0], point[1]] += 1
            selected_points.append(point)
        else:
            unselected.append(point)
    return selected_points

def select_descrete_points_by_distance(points, distance=0):
    """ Remove points within certain range """
    select_points = []
    def _select(point, selected):
        for s in selected:
            d = (point[0]-s[0])**2 + (point[1]-s[1])**2
            if d < distance:
                return False
        return True
    for point in points:
        if _select(point, select_points):
            select_points.append(point)
    return select_points


def match_cos(feature, template):
    feature = feature.permute(1, 2, 0)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    return torch.clamp(cos_similarity, 0, 1)

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


class Tester(object):
    def __init__(self, logger, config):

        dataset_1 = Test_Cephalometric(config['dataset']['pth'], 'Train')
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=1)
        self.config = config

        self.logger = logger
        self.dataset = dataset_1

        # self.sift_landmarks = np.load(config['sift_lm_pth'], allow_pickle=True)
        # assert len(self.sift_landmarks) > 0, f"len {len(self.sift_landmarks)}"

    def test(self, net, oneshot_id=125, draw=False):
        net.eval()
        config = self.config

        self.logger.info(f'ID Oneshot (0~149): {oneshot_id}')
        data = self.dataset.__getitem__(oneshot_id)
        image = data['img']
        # "/home1/quanquan/code/landmark/code/stat_analysis/runs/sift_landmarks2/RawImage/TrainingData/lm/sift_landmarks_001.npy"
        lm_pth = tfilename(config['sift_lm_pth'], 'lm', f'sift_landmarks_{(oneshot_id + 1):03d}.npy')
        # print(lm_pth)
        landmarks_ori = np.load(lm_pth, allow_pickle=True)[0]
        landmarks_ori = np.around(landmarks_ori).astype(np.int)
        n = self.config['num_used_sift_lm']
        if n > 0:
            landmarks_ori = landmarks_ori[:n, :]
        if config['select_policy'] == "select_descrete_points_by_distance":
            landmarks_ori = select_descrete_points_by_distance(landmarks_ori, config['select_distance'])
            print("DEBUG: do select_descrete_points_by_distance! ")

        if oneshot_id in [0,1,2,3,4]:
            print("DEBUG wtf: ", len(landmarks_ori))
            landmarks_ori = landmarks_ori[:100]
            image_pred = visualize(image, landmarks_ori, landmarks_ori, num=len(landmarks_ori))
            image_pred.save(tfilename(config['base']['runs_dir'], f"check_sift_{config['select_distance']}_id{oneshot_id}.png"))
            if oneshot_id == 0:
                print("DEBUG wtf: ", len(landmarks_ori))
                import ipdb; ipdb.set_trace()


        # # TODO: DEBUG-ING
        # return

        image = image.cuda()
        features_tmp = net(image.unsqueeze(0))

        # Depth
        feature_list = dict()
        for id_depth in range(6):
            tmp = list()
            for id_mark, landmark in enumerate(landmarks_ori):
                tmpl_y, tmpl_x = landmark[1] // (2 ** (5 - id_depth)), landmark[0] // (2 ** (5 - id_depth))
                mark_feature = features_tmp[id_depth][0, :, tmpl_y, tmpl_x]
                tmp.append(mark_feature.detach().squeeze().cpu().numpy())
            tmp = np.stack(tmp)
            one_shot_feature = torch.tensor(tmp).cuda()
            feature_list[id_depth] = one_shot_feature

        max_all_list = []
        for data in tqdm(self.dataloader_1):
            img = data['img']
            img = img.cuda()
            features = net(img)
            max_sample_list = []

            pred_landmarks_y, pred_landmarks_x = list(), list()
            for id_mark in range(one_shot_feature.shape[0]):
                cos_lists = []
                max_landmark_list = []
                final_cos = torch.ones_like(img[0, 0]).cuda()
                for id_depth in range(5):
                    cos_similarity = match_cos(features[id_depth].squeeze(), \
                                               feature_list[id_depth][id_mark])
                    cos_similarity = torch.nn.functional.upsample( \
                        cos_similarity.unsqueeze(0).unsqueeze(0), \
                        scale_factor=2 ** (5 - id_depth), mode='nearest').squeeze()
                    final_cos = final_cos * cos_similarity
                    cos_lists.append(cos_similarity)

                for cos in cos_lists:
                    _max = cos.max().cpu().item()
                    max_landmark_list.append(_max)
                _max = final_cos.max().cpu().item()
                # print(_max)
                max_landmark_list.append(_max)
                max_sample_list.append(max_landmark_list)

                final_cos = (final_cos - final_cos.min()) / \
                            (final_cos.max() - final_cos.min())
                cos_lists.append(final_cos)
                chosen_landmark = final_cos.argmax().item()
                pred_landmarks_y.append(chosen_landmark // 384)
                pred_landmarks_x.append(chosen_landmark % 384)

            max_all_list.append(max_sample_list)
            # preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)]

        np.save(tfilename(config['base']['runs_dir'], f'max_list/data_max_list_oneshot_{oneshot_id}.npy'), max_all_list)
        # import ipdb; ipdb.set_trace()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/select/maxsim_sift.yaml')
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)

    config_train = config['training']
    net = UNet_Pretrained(3, non_local=config_train['non_local'], emb_len=config_train['emb_len'])
    probmap = Probmap_np(config)
    net = Wrapper(net=net, probmap=probmap)

    ckpt = "/home1/quanquan/code/landmark/pretrain/" + "cconf_ctr" + f"/model_epoch_1060.pth"
    # BYOL
    # ckpt = '/home1/quanquan/code/landmark/code/runs/byol-2d/byol_std/run4/ckpt/best_model_epoch_50.pth'
    # Barlow
    # ckpt = '/home1/quanquan/code/landmark/code/runs/byol-2d/barlowtwins_ceph/debug/ckpt/best_model_epoch_0.pth'
    # PixelContrast
    # ckpt = '/home1/quanquan/code/landmark/code/runs/ssl/pixel_contra/debugx/ckpt/model_latest.pth'
    print(f"Pretrain {ckpt}")
    config['ckpt'] = ckpt
    assert os.path.exists(ckpt)
    print(f'Load CKPT {ckpt}')

    new_state_dict = {}
    # ckpt = torch.load(ckpt)
    # net.load_state_dict(ckpt)
    net = net.cuda()
    dump_yaml(logger, config)

    tester = Tester(logger, config)
    for i in range(150):
        tester.test(net, oneshot_id=i)

def review():
    pass

if __name__ == '__main__':
    main()

