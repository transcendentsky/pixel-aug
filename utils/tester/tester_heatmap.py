"""
    Basic Tester with regression module

"""
from datasets.eval.eval import Evaluater
from datasets.ceph.ceph_test import Test_Cephalometric
from utils.utils_st import voting
from torch.utils.data import DataLoader
from tqdm import tqdm
from tutils import tfilename
import numpy as np
from torchvision.utils import save_image
from einops import rearrange
from datasets.ceph.ceph_heatmap import Cephalometric
from utils.utils import visualize_landmarks


def heatmap2landmark(heatmap):
    preds = []
    for lm in range(heatmap.shape[0]):
        heatmap_i = heatmap[lm, :, :]
        pred_i = np.unravel_index(heatmap_i.argmax(), heatmap_i.shape)
        # preds.append([pred_i[1], pred_i[0]])
        preds.append([pred_i[0], pred_i[1]])
    return np.array(preds)


def landmark_wh_to_hw(landmarks):
    out = []
    for lm in landmarks:
        out.append([lm[1], lm[0]])
    return np.array(out)


class Tester(object):
    def __init__(self, logger, config, args=None, mode='subtest', get_mre_per_lm=False):

        dataset_1 = Test_Cephalometric(config['dataset']['pth'], mode=mode)
        self.Radius = dataset_1.Radius
        self.config = config
        self.evaluater = Evaluater(logger, [384, 384],
                                   [2400, 1935])
        self.logger = logger

        self.dataset_test = dataset_1
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]
        self.get_mre_per_lm = get_mre_per_lm

    def test(self, model, epoch=0, rank=-1, draw=False, detail=False):
        self.dataloader = DataLoader(self.dataset_test, batch_size=1,
                                     shuffle=False, num_workers=2)
        self.evaluater.reset()
        model.eval()
        ID = 1
        runs_dir = self.config['base']['runs_dir']
        for data in tqdm(self.dataloader, ncols=70):
            if rank != 'cuda' and rank >= 0:
                img = data['img'].to(rank)
            else:
                img = data['img'].cuda()
            landmark_list = data['landmark_list']

            heatmap = model(img)
            pred_landmark = heatmap2landmark(heatmap.cpu().detach().numpy()[0])

            error = self.evaluater.record_new(pred_landmark, landmark_wh_to_hw(landmark_list))
            if detail:
                print(f"debug: id:{ID},  Error:{error.mean()}")
            if draw:
                heatmap = rearrange(heatmap, "n c h w -> c n h w")
                save_image(heatmap, tfilename(runs_dir, f"tmp/heatmap_testset_debug_{ID}.png"))
                # import ipdb; ipdb.set_trace()
                pil_img = visualize_landmarks(img, pred_landmark, landmark_wh_to_hw(landmark_list), num=19)
                pil_img.save(tfilename(runs_dir, f"tmp/testimg_{ID}.png"))
                # np.save("pred.npy", pred_landmark)
                # np.save("lm.npy", landmark_list)
                # import ipdb; ipdb.set_trace()
                # save_image(heatmap)
            # Optional Save viusal results
            # image_pred = visualize(img, pred_landmark, landmark_list)
            # image_pred.save(os.path.join('visuals', str(ID) + '_pred.png'))

            # heatmap = rearrange(heatmap, "x n h w -> n x h w")
            # regression_x = rearrange(regression_x, "x n h w -> n x h w")
            # regression_y = rearrange(regression_y, "x n h w -> n x h w")

            # print("debug: heatmap.shape ", heatmap.shape, regression_y.shape)
            # save_image(heatmap, tfilename(runs_dir, f"heatmap_{ID}.png"))
            # save_image(regression_y, tfilename(runs_dir, f"regy_{ID}.png"))
            # save_image(regression_x, tfilename(runs_dir, f"regx_{ID}.png"))

            ID += 1

        # return {"mre": mre, "sdr": sdr, ...}
        if self.get_mre_per_lm:
            return self.evaluater.cal_metrics_per_lm()
        return self.evaluater.cal_metrics_all()

    def test2(self, model, epoch=0, rank=-1, draw=False):
        config = self.config
        dataset_train = Cephalometric(config['dataset']['pth'], ret_mode="heatmap_only")
        self.dataloader = DataLoader(dataset_train, batch_size=1,
                                     shuffle=False, num_workers=2)
        self.evaluater.reset()
        model.eval()
        ID = 1
        runs_dir = self.config['base']['runs_dir']
        for data in tqdm(self.dataloader, ncols=70):
            img = data['img'].cuda()
            heatmap_gt = data['heatmap']
            # landmark_list = data['landmark_list']

            heatmap = model(img)
            # pred_landmark = heatmap2landmark(heatmap.cpu().detach().numpy()[0])

            # self.evaluater.record(pred_landmark, landmark_list)
            if draw:
                heatmap = rearrange(heatmap, "n c h w -> c n h w")
                save_image(heatmap, tfilename(runs_dir, f"tmp/heatmap_debug_{ID}.png"))
                heatmap_gt = rearrange(heatmap_gt, "n c h w -> c n h w")
                save_image(heatmap_gt, tfilename(runs_dir, f"tmp/gt_debug_{ID}.png"))
                import ipdb; ipdb.set_trace()
            ID += 1
        # return {"mre": mre, "sdr": sdr, ...}
        if self.get_mre_per_lm:
            return self.evaluater.cal_metrics_per_lm()
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


if __name__ == '__main__':
    import torch
    heatmap = torch.zeros((2, 384, 384))

    #
    heatmap[0, 4, 5] = 1
    heatmap[1, 10, 12] = 1
    #
    heatmap[0, 6, 6] = 0.5
    heatmap[1, 8, 6] = 0.5
    lms = heatmap2landmark(heatmap)
    import ipdb; ipdb.set_trace()