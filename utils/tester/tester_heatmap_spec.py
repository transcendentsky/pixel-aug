"""
    Heatmap tester for only one landmark ,

"""
from datasets.eval.eval import Evaluater
from datasets.ceph.ceph_heatmap_spec import Test_Cephalometric
from utils.utils import voting
from torch.utils.data import DataLoader
from tqdm import tqdm
from tutils import tfilename
import numpy as np
from einops import rearrange
# from datasets.ceph.ceph_heatmap import Cephalometric
from utils.utils import visualize_landmarks
from torchvision.utils import save_image


def heatmap2landmark(heatmap):
    preds = []
    for lm in range(heatmap.shape[0]):
        heatmap_i = heatmap[lm, :, :]
        pred_i = np.unravel_index(heatmap_i.argmax(), heatmap_i.shape)
        preds.append(list(pred_i))
    return np.array(preds)


def landmark_wh_to_hw(landmarks):
    out = []
    for lm in landmarks:
        out.append([lm[1], lm[0]])
    return np.array(out)


class Tester(object):
    def __init__(self, logger=None, config=None, args=None, split='subtest', landmark_id=None):
        dataset_1 = Test_Cephalometric(config['dataset']['pth'], split=split, landmark_id=landmark_id)
        self.dataloader = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=2)
        # self.Radius = dataset_1.Radius
        self.config = config
        self.evaluater = Evaluater(logger, [384, 384],
                                       [2400, 1935])
        self.logger = logger
        self.dataset = dataset_1
        self.landmark_id = landmark_id

    def test(self, model, epoch=0, rank=-1, draw=True):
        self.evaluater.reset()
        model.eval()
        runs_dir = self.config['base']['runs_dir']
        ID = 1
        for data in tqdm(self.dataloader, ncols=100):
            img = data['img'].cuda()
            landmark_list = data['landmark_list']
            heatmap = model(img)
            pred_landmark = heatmap2landmark(heatmap.cpu().detach().numpy()[0])
            # self.evaluater.record(pred_landmark, landmark_list)
            # print(pred_landmark,landmark_list)
            error = self.evaluater.record_new(pred_landmark, landmark_wh_to_hw(landmark_list))
            # print("tester debug", error)
            ID += 1
            if draw and ID < 5:
                heatmap = rearrange(heatmap, "n c h w -> c n h w")
                save_image(heatmap, tfilename(runs_dir, f"tmp/heatmap_spec_lm{self.landmark_id}_{ID}.png"))
                # import ipdb; ipdb.set_trace()
                pil_img = visualize_landmarks(img, pred_landmark, landmark_wh_to_hw(landmark_list), num=1)
                pil_img.save(tfilename(runs_dir, f"tmp/testimg_lm{self.landmark_id}_{ID}.png"))

        return self.evaluater.cal_metrics_all()

