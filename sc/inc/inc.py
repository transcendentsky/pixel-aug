import torch
import numpy as np
from models.network_emb_study import UNet_Pretrained

from tutils import trans_init, trans_args, tfilename, save_script, CSVLogger
from tutils.trainer import LearnerModule, Trainer


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.net = UNet()
        self.net_patch = UNet()
        self.label_encoder = Net()
        self.classifier = Net()

    def forward(self, x, **kwargs):

        return

    def training_step(self, data, batch_idx, **kwargs):
        return

    def configure_optimizers(self, **kwargs):
        return

    def save(self, pth, **kwargs):
        return