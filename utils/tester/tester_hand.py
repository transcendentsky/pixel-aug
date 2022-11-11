"""
    Basic Tester with regression module

"""
from datasets.eval.eval_hand import Evaluater
from datasets.hand.hand_basic import HandXray
from utils.utils_st import voting
from torch.utils.data import DataLoader
from tqdm import tqdm
from tutils import tfilename
import numpy as np
from utils.utils import visualize


class Tester(object):
    def __init__(self, logger, config, args=None, split='subtest'):

        dataset = HandXray(pathDataset=config['dataset']['pth'], split=split)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=1,
                                     shuffle=False, num_workers=2)
        self.Radius = dataset.Radius
        self.config = config
        self.evaluater = Evaluater(logger, [384, 384])
        self.logger = logger

        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]

    def test(self, model, epoch=0, rank=-1):
        self.evaluater.reset()
        model.eval()
        ID = 1
        for data in tqdm(self.dataloader, ncols=70):
            if rank != 'cuda' and rank >= 0:
                img = data['img'].to(rank)
            else:
                img = data['img'].cuda()
            landmark_list = data['landmark_list']
            heatmap, regression_y, regression_x = model(img)

            # Vote for the final accurate point
            pred_landmark, votings = voting( \
                heatmap, regression_y, regression_x, self.Radius, get_voting=True) # pred: [h, w]

            # self.evaluater.record(pred_landmark, landmark_list)
            self.evaluater.record_hand(pred_landmark, landmark_list)

            # Optional Save viusal results
            image_pred = visualize(img, pred_landmark, landmark_list, num=37)
            image_pred.save(tfilename(self.config['base']['runs_dir'], 'visuals', str(ID) + '_pred.png'))

            ID += 1

        # return {"mre": mre, "sdr": sdr, ...}
        return self.evaluater.cal_metrics()



    def dump_pseudo_dataset(self, model, iteration=1):
        model.eval()
        ID = 1

        dataset = TestHandXray(self.config['dataset']['pth'], mode='Test')
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
        return self.evaluater.cal_metrics()
