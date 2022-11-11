from .eval import Evaluater
from datasets.ceph_test import Test_Cephalometric
from .utils import voting
from torch.utils.data import DataLoader
from tqdm import tqdm
from tutils import tfilename
import numpy as np


class Tester(object):
    def __init__(self, logger, config, args=None, mode='subtest'):

        dataset_1 = Test_Cephalometric(config['dataset']['pth'], mode=mode)
        self.dataloader = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=2)
        self.Radius = dataset_1.Radius
        self.config = config
        self.evaluater = Evaluater(logger, [384, 384],
                                       [2400, 1935])
        self.logger = logger

        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]

    def test(self, model, epoch=0):
        self.evaluater.reset()
        model.eval()
        ID = 1
        voting_list = [[] for _ in range(19)]
        error_list = [[] for _ in range(19)]
        for data in tqdm(self.dataloader, ncols=70):
            img = data['img'].cuda()
            landmark_list = data['landmark_list']

            heatmap, regression_y, regression_x = model(img)

            # gray_to_PIL(heatmap[0][1].cpu().detach()) \
            #     .save(os.path.join('visuals', str(ID) + '_heatmap.png'))
            # Vote for the final accurate point
            pred_landmark, votings = voting( \
                heatmap, regression_y, regression_x, self.Radius, get_voting=True)

            self.evaluater.record(pred_landmark, landmark_list)

            # Optional Save viusal results
            # image_pred = visualize(img, pred_landmark, landmark_list)
            # image_pred.save(os.path.join('visuals', str(ID) + '_pred.png'))

            ID += 1

        # return {"mre": mre, "sdr": sdr, ...}
        return self.evaluater.cal_metrics_all()

    def debug(self, model):
        print("DEBUG")
        model.eval()
        self.evaluater.reset()
        for data in self.dataloader:
            print(data['name'])
            img = data['img'].cuda()
            landmark_list = data['landmark_list']
            heatmap, regression_y, regression_x = model(img, return_features=True)
            break
        print("DEBUG")


    def dump_pseudo_dataset(self, model, iteration=1):
        model.eval()
        ID = 1

        dataset = Test_Cephalometric(self.config['dataset']['pth'], mode='Train')
        trainloader = DataLoader(dataset, batch_size=1,
                                       shuffle=False, num_workers=2)

        for i, data in tqdm(enumerate(trainloader), ncols=100):
            img = data['img'].cuda()
            landmark_list = data['landmark_list']

            heatmap, regression_y, regression_x = model(img)
            pred_landmark, votings = voting( \
                heatmap, regression_y, regression_x, self.Radius, get_voting=True)
            self.evaluater.record(pred_landmark, landmark_list)
            pred_landmark = np.array(pred_landmark).transpose((1, 0))
            np.save(tfilename(self.config["runs_dir"], "pseudo_labels", f"iter_{iteration}", f"{ID}.npy"), np.array(pred_landmark))
            if i <= 0:
                self.logger.warn(f" shape {np.array(pred_landmark).shape}")
                print("[] Np.save ", f"iter_{iteration}/" + f"{ID}.npy")
            ID += 1
        return self.evaluater.cal_metrics_all()
