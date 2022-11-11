"""
    Basic Tester without regression module
"""
from .eval import Evaluater
from datasets.ceph_test import Test_Cephalometric
from .utils import voting
from torch.utils.data import DataLoader
from tqdm import tqdm
from tutils import tfilename
import numpy as np


def heatmap2landmark(heatmap):
    preds = []
    for lm in range(heatmap.shape[0]):
        heatmap_i = heatmap[lm, :, :]
        pred_i = np.unravel_index(heatmap_i.argmax(), heatmap_i.shape)
        preds.append(list(pred_i))
    return np.array(preds)


class Tester(object):
    def __init__(self, logger=None, config=None, args=None, mode='subtest'):

        dataset_1 = Test_Cephalometric(config['dataset']['pth'], mode=mode)
        self.dataloader = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=2)
        # self.Radius = dataset_1.Radius
        self.config = config
        self.evaluater = Evaluater(logger, [384, 384],
                                       [2400, 1935])
        self.logger = logger

        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['special']['num_landmarks'])]

    def test(self, model, epoch=0, rank=-1):
        self.evaluater.reset()
        model.eval()
        ID = 1
        for data in tqdm(self.dataloader, ncols=100):
            if rank >= 0:
                img = data['img'].to(rank)
            else:
                img = data['img'].cuda()
            landmark_list = data['landmark_list']
            heatmap = model(img)
            pred_landmark = heatmap2landmark(heatmap.cpu().detach().numpy()[0])

            # pred_landmark = np.unravel_index(heatmap.argmax(), heatmap.shape)

            self.evaluater.record(pred_landmark, landmark_list)

            # Optional Save viusal results
            # image_pred = visualize(img, pred_landmark, landmark_list)
            # image_pred.save(os.path.join('visuals', str(ID) + '_pred.png'))

            ID += 1

        return self.evaluater.cal_metrics_all()

    # def dump_pseudo_dataset(self, model, iteration=1):
    #     model.eval()
    #     ID = 1
    #
    #     dataset = Test_Cephalometric(self.config['dataset']['pth'], mode='Train')
    #     trainloader = DataLoader(dataset, batch_size=1,
    #                                    shuffle=False, num_workers=2)
    #
    #     for i, data in tqdm(enumerate(trainloader), ncols=100):
    #         img = data['img'].cuda()
    #         landmark_list = data['landmark_list']
    #
    #         heatmap, regression_y, regression_x = model(img)
    #         pred_landmark, votings = voting( \
    #             heatmap, regression_y, regression_x, self.Radius, get_voting=True)
    #         self.evaluater.record(pred_landmark, landmark_list)
    #         pred_landmark = np.array(pred_landmark).transpose((1, 0))
    #         np.save(tfilename(self.config["runs_dir"], "pseudo_labels", f"iter_{iteration}", f"{ID}.npy"), np.array(pred_landmark))
    #         if i <= 0:
    #             self.logger.warn(f" shape {np.array(pred_landmark).shape}")
    #             print("[] Np.save ", f"iter_{iteration}/" + f"{ID}.npy")
    #         ID += 1
    #     return self.evaluater.cal_metrics_all()
