import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
from pathlib import Path

def make_dir(pth):
    dir_pth = Path(pth)
    if not dir_pth.exists():
        dir_pth.mkdir()
    return pth


class Evaluater(object):
    def __init__(self, logger, size, original_size, tag='paper_figure'):
        self.pixel_spaceing = 1.0
        self.tag = tag
        make_dir(tag)
        self.tag += '/'

        # self.logger = logger
        self.logger = None
        self.scale_rate_y = original_size[0] / size[0]
        self.scale_rate_x = original_size[1] / size[1]

        self.RE_list = list()

        self.recall_radius = [2, 2.5, 3, 4]  # 2mm etc
        self.recall_rate = list()

        self.mode_list = [0, 1, 2, 3]
        self.mode_dict = {0: "Iterative FGSM", 1: "Adaptive Iterative FGSM", \
                          2: "Adaptive_Rate", 3: "Proposed"}

        self.best_mre = 100.0

    def info(self, msg, *args, **kwargs):
        pass
        # if self.logger is not None:
        #     self.logger.info(msg)
        # else:
        #     print(msg, *args, **kwargs)

    def reset(self):
        self.RE_list.clear()

    def record(self, pred, landmark):
        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)
        return Radial_Error

    def cal_metrics(self, return_sdr=False):
        # calculate MRE SDR
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        # self.info(Mean_RE_channel)
        mre = Mean_RE_channel.mean()
        # self.info("ALL MRE {}".format(mre))

        sdr_dict = {}
        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            sdr_dict[f"SDR {radius}"] = shot * 100 / total
            # self.info("ALL SDR {}mm  {}".format\
            #                      (radius, shot * 100 / total))
        if return_sdr:
            return {'mre':mre, **sdr_dict}
        return {'mre':mre}

    def cal_metrics_all(self):
        return self.cal_metrics(return_sdr=True)

