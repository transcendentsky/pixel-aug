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
    cos_sim = cosfn(rearrange(template, "1 c h w-> 1 c h w"), rearrange(feature, "1 c h w-> 1 c h w"))
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim

def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc=None):
    scale = 2 ** (5-ii)
    gt_values = match_inner_product(raw_fea_list[ii], crop_fea_list[ii])
    gt_values = torch.nn.functional.upsample( \
        gt_values.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze()
    gt_values = gt_values.detach().cpu().numpy()
    if raw_loc is not None:
        vv = [gt_values[loc[1].item(), loc[0].item()] for loc in raw_loc]
        # import ipdb; ipdb.set_trace()
        return np.array(vv)
    # print("gtvalues", gt_values)
    # print(gt_values)
    return rearrange(gt_values, "h w -> (h w)")


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
                 cj_brightness=0.8,
                 cj_contrast=0.6,
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
                                      cj_brightness=cj_brightness, cj_contrast=cj_contrast,
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

    @staticmethod
    def training_step(net, net_patch, data, use_lm=False, temperature=None, **kwargs):
        raw_imgs = data['img1'].cuda()
        crop_imgs = data['img2'].cuda()
        raw_loc = data['raw_loc']
        chosen_loc = raw_loc
        entr_value = data['entr_value']
        entr_map0 = data['entr_map']
        landmarks = data['landmarks'] if use_lm else None
        if use_lm:
            entr_map = np.array([entr_map0[0][loc[1].item(), loc[0].item()].item() for loc in landmarks])
        else:
            entr_map = rearrange(entr_map0, " 1 h w -> (h w)")
        # import ipdb; ipdb.set_trace()
        raw_fea_list = net(raw_imgs)
        crop_fea_list = net_patch(crop_imgs)

        gt_value_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, landmarks)
        gt_value_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, landmarks)
        gt_value_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, landmarks)
        gt_value_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, landmarks)
        gt_value_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, landmarks)
        torchvision_save(raw_imgs[0], "./tmp/test_gtv1.png")
        torchvision_save(crop_imgs[0], "./tmp/test_gtv2.png")
        gtv = gt_value_0 * gt_value_1 * gt_value_2 * gt_value_3 * gt_value_4
        # dd = {'gtv': gtv, 'gtv_0': gt_value_0, 'gtv_1': gt_value_1, 'gtv_2': gt_value_2, 'gtv_3': gt_value_3,
        #              'gtv_4': gt_value_4, "entr_map": entr_map}
        if temperature is None:
            dd = {'gtv': gtv, "entr_map": entr_map}
            return dd
        else:
            prob_map = entr_map ** temperature
            prob_map /= prob_map.mean()
            gtv2 = gtv * prob_map.detach().cpu().numpy()
            dd = {'gtv': gtv, "entr_map": entr_map, "gtv2": gtv2}
            return dd
        # import ipdb; ipdb.set_trace()
        # print("debug , ", dd)

    def test_func(self, learner, epoch=None, rank='cuda', oneshot_id=-1, draw=False):
        nearby = self.config['special']['nearby']
        self.recorder.clear()
        learner.eval()
        net = learner.net
        net_patch = learner.net_patch if self.use_both_encoder else net
        for i, data in enumerate(self.dataloader):
            print("Processing ", i, end='\r')
            res = self.training_step(net, net_patch, data, nearby)
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
            if i > 100:
                break
        print("Ready to cal_metrics()")
        d, length = self.recorder.cal_metrics2()
        # to interpreted list
        return d, length

    def test_func3(self, learner, epoch=None, rank='cuda', oneshot_id=-1, draw=False):
        """ for showing increased variance """
        self.recorder.clear()
        learner.eval()
        net = learner.net
        net_patch = learner.net_patch if self.use_both_encoder else net
        for i, data in enumerate(self.dataloader):
            print("Processing ", i, end='\r')
            res = self.training_step(net, net_patch, data, temperature=0.3)
            self.recorder.record(res)
            if i > 100:
                break
        print("Ready to cal_metrics()")
        d, length = self.recorder.cal_metrics()
        # to interpreted list
        gtvs1 = [v[0] for v in d]
        gtvs3 = [v[2] for v in d]
        return gtvs1, gtvs3, length

    def test_func4(self, learner, epoch=None, rank='cuda', oneshot_id=-1, draw=False):
        """ for showing increased variance """
        self.recorder.clear()
        learner.eval()
        net = learner.net
        net_patch = learner.net_patch if self.use_both_encoder else net
        for i, data in enumerate(self.dataloader):
            print("Processing ", i, end='\r')
            res = self.training_step(net, net_patch, data, temperature=0.3)
            self.recorder.record(res)
            if i > 100:
                break
        print("Ready to cal_metrics()")
        d = self.recorder.cal_metrics3()
        return d


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
        temp = rearrange(temp, "b c p -> (b p) c")
        listlist = []
        for i in range(7):
            list1 = temp.copy()[(temp[:, 1]>=i) & (temp[:, 1]<(i+1))]
            listlist.append(list1)

        list_mean = [l.mean(axis=0) for l in listlist]
        length = [len(l) for l in listlist]
        return list_mean, length

    def cal_metrics2(self):
        temp = np.array(self.loss_list)
        num = temp.shape[0]
        temp = temp[:, 0, :]
        mean = temp.mean(axis=0)
        length = [num] * 19
        # import ipdb; ipdb.set_trace()
        return mean, length

    def cal_metrics3(self):
        len_record = len(self.loss_list)
        temp = np.array(self.loss_list)
        # import ipdb; ipdb.set_trace()
        temp = rearrange(temp, "b c p -> (b p) c")

        entr = temp[:, 1]**0.3
        value = temp[:, 0]
        return (entr * value).mean() / entr.mean()

        # listlist = []
        # for i in range(7):
        #     list1 = temp.copy()[(temp[:, 1]>=i) & (temp[:, 1]<(i+1))]
        #     listlist.append(list1)
        #
        # list_mean = [l.mean(axis=0) for l in listlist]
        # length = [len(l) for l in listlist]
        # return list_mean, length

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
