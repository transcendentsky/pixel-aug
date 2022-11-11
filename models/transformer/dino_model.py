import torch
from torch import nn
import torch.nn.functional as F

from .vit2 import VisionTransformer
from .decoder import MaskTransformer

from .utils import padding, unpadding
# from timm.models.helpers import load_pretrained, load_custom_pretrained
# from timm.models.vision_transformer import default_cfgs
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _create_vision_transformer
# from timm.models.layers import trunc_normal_
# from tutils import print_dict
# from .utils import checkpoint_filter_fn


class DINO(nn.Module):
    def __init__(self, config):
        super(DINO, self).__init__()
        self.config = config

        self.student = VisionTransformer(patch_size=config['patch_size'], pretrained=False, drop_path_rate=config['drop_path_rate'])
        self.teacher = VisionTransformer(patch_size=config['patch_size'])
        # self.n_cls = config['n_cls']
        self.patch_size = 96 # encoder.patch_size  #
        self.encoder = None # encoder
        self.decoder = None # decoder

        self.model_cfg = config['model_cfg']
        self.decoder_cfg = config['decoder_cfg']
        self.n_classes = config['dataset']['n_cls'] # config['special']['num_landmarks']
        self.regress_module = config['special']['regress']

        self.build()

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))
        if not self.regress_module:
            return masks

        heatmap = F.sigmoid(masks[:, :self.n_classes, :, :])
        regression_x = masks[:, self.n_classes:2 * self.n_classes, :, :]
        regression_y = masks[:, 2 * self.n_classes:, :, :]
        return heatmap, regression_x, regression_y