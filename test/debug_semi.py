import torch
import torchvision
import numpy as np
import os
from tutils import torchvision_save


def test_eval():
    from datasets.eval.eval import Evaluater
    evaluater = Evaluater(None, [384, 384], [2400, 1935])

    pred = np.load("../self-train-qs/rand_pred.npy")
    landmark = np.load("../self-train-qs/rand_lm.npy")

    print(pred)
    print(landmark)
    evaluater.record_old(pred, landmark)
    res = evaluater.cal_metrics_all()
    print(res)


def test_model():
    tproj_data = torch.load("/home1/quanquan/code/landmark/code/tproj/debug_data.pt")
    selft_data = torch.load("/home1/quanquan/code/landmark/code/self-train-qs/debug_data.pt")
    img, mask, offset_x, offset_y, landmark_list, heatmap, regression_y, regression_x = tproj_data

    img2, mask2, offset_x2, offset_y2, landmark_list2, heatmap2, regression_y2, regression_x2 = selft_data
    import ipdb; ipdb.set_trace()
    pass


if __name__ == '__main__':
    test_model()