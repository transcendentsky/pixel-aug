
import numpy as np
import csv
from collections.abc import Iterable
from tutils import tdir, tfilename
# from utils import make_dir

WRIST_WIDTH = 50

def radial(pt1, pt2, factor=1):
    if not isinstance(factor, Iterable):
        factor = [factor]*len(pt1)
    rad = sum(((i-j)*s)**2 for i, j, s in zip(pt1, pt2, factor))**0.5
    assert rad > 0 , f"Got {rad}"
    return rad

class Evaluater(object):
    def __init__(self, logger, size):
        # self.pixel_spaceing =
        self.logger = logger
        self.RE_list = list()

        self.recall_radius = [2, 2.5, 3, 4, 10]  # 2mm etc
        self.recall_rate = list()

        # Just for hand dataset
        self.size = size  # original size is not fixed for hand dataset, we calculate it in realtime
        # self.scale_rate_y = 3900 / 384
        # self.scale_rate_x = 2700 / 384

    def reset(self):
        self.RE_list.clear()

    # def record(self, pred, landmark):
    #     c = pred[0].shape[0]
    #     diff = np.zeros([c, 2], dtype=float)  # y, x
    #     for i in range(c):
    #         diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y
    #         diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
    #     Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
    #     # Radial_Error *= self.pixel_spaceing
    #     self.RE_list.append(Radial_Error)
    #     return None
    def record_hand(self, pred, landmark):
        """
        :param pred: [h, w], [h,w]
        :param landmark: [w,h], [w,h]
        :return:
        """
        pred = np.array(pred)
        landmark = np.array(landmark)
        scale_rate = WRIST_WIDTH / radial(landmark[0], landmark[4])
        c = pred[0].shape[0]
        assert c == 37, f"Got {c}"
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[1][i] - landmark[i][0]) * scale_rate
            diff[i][1] = abs(pred[0][i] - landmark[i][1]) * scale_rate
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        self.RE_list.append(Radial_Error)


    def record_hand_old2(self, pred, landmark, img_shape=None):
        """
        Function for testing hand dataset, (due to the different "img_shape")
        pred: [w, h], [w, h]
        landmark: [w, h], [w, h]
        """
        # scale_rate_y = img_shape[0] / self.size[0]
        # scale_rate_x = img_shape[1] / self.size[1]
        # raise NotImplementedError
        pred = np.array(pred)
        landmark = np.array(landmark)
        scale_rate = WRIST_WIDTH / radial(landmark[0], landmark[4])
        c = pred[0].shape[0]
        assert c == 37, f"Got {c}"
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][0]) * scale_rate
            diff[i][1] = abs(pred[1][i] - landmark[i][1]) * scale_rate
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        # print("Extreme bug ", scale_rate, Radial_Error, landmark[0], pred[:,0], landmark[4], pred[:,4])
        # Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)

    def record_hand_old(self, pred, landmark):
        """
        :param pred: [w, h], [w, h]
        :param landmark: [w, h], [w, h]
        :return:
        """
        pred = np.array(pred)
        landmark = np.array(landmark)
        assert pred.shape == (37, 2), f"Got {pred.shape}, should be (37, 2)"
        scale_rate = WRIST_WIDTH / radial(landmark[0], landmark[4])
        c = len(pred)
        assert c == 37, f"Got {c}"
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[i][0] - landmark[i][0]) * scale_rate
            diff[i][1] = abs(pred[i][1] - landmark[i][1]) * scale_rate
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        # Radial_Error *= self.pixel_spaceing
        # print("Extreme bug ", scale_rate, Radial_Error, landmark[0], landmark[4])
        self.RE_list.append(Radial_Error)

    def cal_metrics(self, debug=False):
        # calculate MRE SDR
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        # self.logger.info(Mean_RE_channel)
        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(Mean_RE_channel.tolist())
        mre = Mean_RE_channel.mean()
        # self.log("ALL MRE {}".format(mre))

        sdr_dict = {}
        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            # self.log("ALL SDR {}mm  {}".format \
            #                      (radius, shot * 100 / total))
            sdr_dict[f"SDR {radius}"] = shot * 100 / total

        if debug:
            import ipdb; ipdb.set_trace()
        ret_dict = {'mre': mre}
        ret_dict = {**ret_dict, **sdr_dict}
        return ret_dict

    def log(self, msg, *args, **kwargs):
        if self.logger is None:
            print(msg, *args, **kwargs)
        else:
            self.logger.info(msg, *args, **kwargs)

