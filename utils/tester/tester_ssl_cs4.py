"""
    Basic Tester with regression module

"""
import torch
from datasets.eval.eval import Evaluater
from datasets.ceph.ceph_ssl import Test_Cephalometric
from utils.utils_st import voting
from torch.utils.data import DataLoader
from tqdm import tqdm
from tutils import tfilename
import numpy as np
from einops import rearrange, repeat


class Tester(object):
    def __init__(self, logger, config, args=None, mode='subtest'):

        dataset_1 = Test_Cephalometric(config['dataset']['pth'], mode=mode, return_mode="more")
        self.dataloader = DataLoader(dataset_1, batch_size=1,
                                     shuffle=False, num_workers=2)
        self.Radius = dataset_1.Radius
        self.config = config
        self.evaluater = Evaluater(logger, [384, 384],
                                   [2400, 1935])
        self.logger = logger

        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]

    def test(self, net, epoch=0, rank=-1):
        self.evaluater.reset()
        net.eval()
        ID = 1
        voting_list = [[] for _ in range(19)]
        error_list = [[] for _ in range(19)]
        if rank >= 0:
            raise NotImplementedError

        for data in tqdm(self.dataloader, ncols=70):
            raw_imgs = data['raw_imgs'].cuda()
            crop_imgs = data['crop_imgs'].squeeze().cuda()
            raw_loc = data['landmark_list']
            chosen_loc = data['chosen_loc']
            chosen_loc = torch.LongTensor(chosen_loc)
            num_landmark = crop_imgs.shape[0]
            # import ipdb; ipdb.set_trace()

            heatmap_list = []
            regression_x_list = []
            regression_y_list = []
            for i in range(num_landmark):
                # import ipdb; ipdb.set_trace()
                heatmap, regression_y, regression_x = net(raw_imgs, crop_imgs[i].unsqueeze(0), chosen_loc[i].unsqueeze(0))
                # To convert from (19, 1, 384, 384) to (1, 19, 384, 384)
                # heatmap = rearrange(heatmap, "c n h w -> n c h w")
                # regression_y = rearrange(regression_y, "c n h w -> n c h w")
                # regression_x = rearrange(regression_x, "c n h w -> n c h w")

                # gray_to_PIL(heatmap[0][1].cpu().detach()) \
                #     .save(os.path.join('visuals', str(ID) + '_heatmap.png'))
                # Vote for the final accurate point
                # import ipdb; ipdb.set_trace()
                heatmap_list.append(heatmap)
                regression_x_list.append(regression_x)
                regression_y_list.append(regression_y)
            heatmap = torch.cat(heatmap_list, dim=1)
            regression_x = torch.cat(regression_x_list, dim=1)
            regression_y = torch.cat(regression_y_list, dim=1)
            assert heatmap.shape == (1,19,384,384), f"Got {heatmap.shape}"

            pred_landmark, votings = voting( \
                heatmap, regression_y, regression_x, self.Radius, get_voting=True)


            landmark_list = raw_loc
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
