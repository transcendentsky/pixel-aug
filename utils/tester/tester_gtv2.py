import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from PIL import Image, ImageDraw, ImageFont
from datasets.ceph.ceph_ssl import Test_Cephalometric
from utils.utils import visualize
from datasets.ceph.ceph_ssl_adapm import Cephalometric
from tutils import tfilename, tdir, torchvision_save
from tutils.tutils.recorder import Recorder
from einops import rearrange


cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, nearby=None):
    b, w, h = cos_map.shape
    total_loss = list()
    gt_values_to_record = []
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
        gt_values_to_record.append(gt_value.clone().detach().log().cpu())
        total_loss.append(id_loss)
    gt_values_to_record = torch.stack(gt_values_to_record).mean()
    return torch.stack(total_loss).mean(), gt_values_to_record


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    # import ipdb; ipdb.set_trace()
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss, gt_values = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss, gt_values


def ce_loss_0(cos_map, gt_x, gt_y, nearby=None):
    b, w, h = cos_map.shape
    total_loss = list()
    gt_values_to_record = []
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
        gt_values_to_record.append(gt_value.clone().detach().log().cpu())
        total_loss.append(id_loss)
    gt_values_to_record = torch.stack(gt_values_to_record).mean()
    return torch.stack(total_loss), gt_values_to_record


def _tmp_loss_func_0(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    # import ipdb; ipdb.set_trace()
    tmpl_feature = torch.stack([crop_fea_list[ii].cpu()[0, :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii].cpu(), tmpl_feature)  # shape [8,12,12]
    loss, gt_values = ce_loss_0(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss, gt_values


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
                 default_oneshot_id=114,
                 upsample="nearest",
                 collect_sim=False,
                 use_both_encoder=False,
                 **kwargs):
        self.split = split
        self.retfunc = retfunc
        self.config = config
        self.args = args
        self.model = net
        self.default_oneshot_id = default_oneshot_id
        self.upsample = upsample
        self.collect_sim = collect_sim
        self.use_both_encoder = use_both_encoder
        print("Warning! trying collecting sims! ")

        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]
        print("Tester settings: upsample:", self.upsample)

        # testset = Test_Cephalometric(
        #     "/home1/quanquan/code/landmark/code/runs/ssl/dump_label_from_ssl/ssl_baseline/pseudo_labels_init",
        #     mode="pseudo", pre_crop=False)
        testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
        all_landmarks = []
        for i in range(len(testset)):
            landmark_list = testset.ref_landmarks(i)
            all_landmarks.append(landmark_list)

        dataset_train = Cephalometric(config['dataset']['pth'], patch_size=384,
                                      mode=split, retfunc=3, hard_select=True, multi_output=False,
                                      num_repeat=1000,
                                      cj_brightness=0.8, cj_contrast=0.6,
                                      runs_dir=config['base']['runs_dir'],
                                      entr_map_dir='/home1/quanquan/datasets/Cephalometric/entr1/train/',
                                      use_prob=False,
                                      return_entr=True)
        dataset_train.entr_map_from_image(temperature=1)
        assert len(dataset_train) >= 1000, f"Got {len(dataset_train)}"

        self.dataloader = DataLoader(dataset_train, batch_size=1,
                                       shuffle=True, num_workers=config['training']['num_workers'])
        self.recorder = MyRecorder()

    def test(self, *args, **kwargs):
        if self.retfunc == 1:
            return self.test_func(*args, **kwargs)
        elif self.retfunc == 2:
            return self.test_func2(*args, **kwargs)
        elif self.retfunc == 3:
            return self.test_func3(*args, **kwargs)
        else:
            raise NotImplementedError

    def training_step(self, net, net_patch, data, use_lm=False, temperature=None, **kwargs):
        raw_imgs = data['img1'].cuda()
        crop_imgs = data['img2'].cuda()
        raw_loc = data['raw_loc']
        chosen_loc = raw_loc
        entr_value = data['entr_value']
        entr_map0 = data['entr_map']
        # landmarks = torch.stack(data['landmarks']).detach().cpu().numpy() if use_lm else None
        # import ipdb; ipdb.set_trace()
        raw_fea_list = net(raw_imgs)
        crop_fea_list = net_patch(crop_imgs)

        nearby = self.config['special']['nearby']
        if not use_lm:
            loss_0, gt_value_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
            loss_1, gt_value_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_2, gt_value_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_3, gt_value_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_4, gt_value_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            # torchvision_save(raw_imgs[0], "./tmp/test_gtv1.png")
            # torchvision_save(crop_imgs[0], "./tmp/test_gtv2.png")
            loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
            gtv = gt_value_0 * gt_value_1 * gt_value_2 * gt_value_3 * gt_value_4
            loss_dict = {'loss': loss, 'entr_value': entr_value}
            # print(loss_dict)
            # import ipdb; ipdb.set_trace()
            return loss_dict
        else:
            landmarks = np.array([[lm[1].item(), lm[0].item()] for lm in data['landmarks']])
            entr_list = np.array([entr_map0[0][loc[1].item(), loc[0].item()].item() for loc in landmarks])
            # for lm in landmarks:
            raw_loc = landmarks
            chosen_loc = landmarks
            loss_0, gt_value_0 = _tmp_loss_func_0(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
            loss_1, gt_value_1 = _tmp_loss_func_0(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_2, gt_value_2 = _tmp_loss_func_0(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_3, gt_value_3 = _tmp_loss_func_0(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss_4, gt_value_4 = _tmp_loss_func_0(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
            loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
            gtv = gt_value_0 * gt_value_1 * gt_value_2 * gt_value_3 * gt_value_4
            # import ipdb; ipdb.set_trace()
            loss = loss.detach().numpy()
            return {'loss': loss, 'entr_value': entr_list}


    def test_func(self, learner, epoch=None, rank='cuda', oneshot_id=-1, draw=False):
        self.recorder.clear()
        learner.eval()
        net = learner.net
        net_patch = learner.net_patch if self.use_both_encoder else net
        for i, data in enumerate(self.dataloader):
            res = self.training_step(net, net_patch, data, use_lm=False)
            print("Processing ", i, end='\r')
            self.recorder.record(res)
            if i > 100:
                break
        print("Ready to cal_metrics()")
        d, length = self.recorder.cal_metrics()

        # to interpreted list
        gtvs = [v[0] for v in d]
        return gtvs, length

    def test_func2(self, learner, epoch=None, rank='cuda', oneshot_id=-1, draw=False):
        nearby = self.config['special']['nearby']
        self.recorder.clear()
        learner.eval()
        net = learner.net
        net_patch = learner.net_patch if self.use_both_encoder else net
        for i, data in enumerate(self.dataloader):
            print("Processing ", i, end='\r')
            res = self.training_step(net, net_patch, data, use_lm=True)
            self.recorder.record(res)
            if i > 1000:
                break
        print("Ready to cal_metrics()")
        d, length = self.recorder.cal_metrics2()
        # to interpreted list
        return d, length

class MyRecorder(Recorder):
    def __init__(self,  *args, **kwargs):
        # self.thres = thres
        super(MyRecorder, self).__init__(*args, **kwargs)

    def record(self, loss:dict) -> None:
        assert type(loss) == dict, f"Got {loss}"
        # print("debug record", loss)
        if self.loss_keys is None:
            self.loss_list = []
            self.loss_keys = loss.keys()
        l_list = []
        for key, value in loss.items():
            if type(value) == torch.Tensor:
                l_list.append(value.detach().cpu().numpy())
            elif type(value) in [str, bool]:
                pass
            elif type(value) in [np.ndarray, np.float64, np.float32, int, float]:
                l_list.append(value)
            else:
                print("debug??? type Error? , got ", type(value))
                print("debug??? ", key, value)
                l_list.append(float(value))
        self.loss_list.append(l_list)

    def cal_metrics(self):
        len_record = len(self.loss_list)
        temp = np.array(self.loss_list)
        # import ipdb; ipdb.set_trace()
        # temp = rearrange(temp, "b c p -> (b p) c")
        listlist = []
        list_mean = []
        length = []
        thres = np.arange(0, 8)
        thres[-1] = 999
        for i in range(7):
            list1 = temp.copy()[(temp[:, 1]>=thres[i]) & (temp[:, 1]<thres[i+1])]
            # import ipdb; ipdb.set_trace()
            list1_mean_value = [list1.mean(), list1.mean()] if len(list1) == 0 else list1.mean(axis=0) # ext
            # listlist.append(list1)
            list_mean.append(list1_mean_value)
            length.append(len(list1))

        # list_mean = [l.mean(axis=0) for l in listlist]
        # length = [len(l) for l in listlist]
        return list_mean, length

    def cal_metrics2(self):
        temp = np.array(self.loss_list)
        num = temp.shape[0]
        temp = temp[:, 0, :]
        mean = temp.mean(axis=0)
        length = [num] * 19
        # import ipdb; ipdb.set_trace()
        return mean, length

    # def cal_metrics2(self):
        # if self.reduction == "mean":
        #     res = temp.mean(axis=0)
        # elif self.reduction == "sum":
        #     res = temp.sum(axis=0)
        # else:
        #     raise ValueError
        # # print("debug mean", mean, temp, self.loss_keys)
        #
        # if self.loss_keys is None:
        #     return None
        # else:
        #     _dict = {k: res[i] for i, k in enumerate(self.loss_keys)}
        #     _dict['num'] = len_record
        #     return _dict
