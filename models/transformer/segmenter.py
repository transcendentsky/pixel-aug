import torch
from torch import nn
import torch.nn.functional as F

from .vit import VisionTransformer
from .decoder import MaskTransformer

from .utils import padding, unpadding
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
from timm.models.layers import trunc_normal_
from tutils import print_dict
from .utils import checkpoint_filter_fn
from torchvision.utils import save_image
from einops import rearrange


class Segmenter(nn.Module):
    def __init__(self, config):
        super(Segmenter, self).__init__()
        self.config = config
        # self.n_cls = config['n_cls']
        self.patch_size = 16 # encoder.patch_size  #
        self.encoder = None # encoder
        self.decoder = None # decoder

        self.model_cfg = config['model_cfg']
        self.decoder_cfg = config['decoder_cfg']
        self.n_classes = config['dataset']['n_cls'] # config['special']['num_landmarks']
        self.regress_module = config['special']['regress']

        self.build()

    def forward_to_masks(self, im, visualize=False, **kwargs):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        masks = self.decoder(x, (H, W))
        return masks

    def forward(self, im, visualize=False, **kwargs):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))
        if visualize:
            print("shape: ", masks.shape)
            save_image(masks[0].unsqueeze(1), "./tmp/masks_0.png")

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        if visualize:
            print("shape: ", masks.shape)
            save_image(masks[0].unsqueeze(1), "./tmp/masks_1.png")

        if not self.regress_module:
            # return masks
            return F.sigmoid(masks)

        heatmap = F.sigmoid(masks[:, :self.n_classes, :, :])
        regression_x = masks[:, self.n_classes:2 * self.n_classes, :, :]
        regression_y = masks[:, 2 * self.n_classes:, :, :]
        return heatmap, regression_x, regression_y

    def build(self):
        self.encoder = self.create_vit(self.model_cfg)
        self.decoder = self.create_decoder(self.encoder, self.decoder_cfg)
        # model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    def create_vit(self, model_cfg):
        model_cfg["n_cls"] = 1000
        mlp_expansion_ratio = 4
        model_cfg['d_ff'] = mlp_expansion_ratio * model_cfg["d_model"]
        # get model
        model = VisionTransformer(**model_cfg)

        backbone = model_cfg['backbone']
        if backbone in default_cfgs:
            default_cfg = default_cfgs[backbone]
        else:
            raise NotImplementedError(f"Backbone: {backbone}")
            # default_cfg = dict(
            #     pretrained=False,
            #     num_classes=1000,
            #     drop_rate=0.0,
            #     drop_path_rate=0.0,
            #     drop_block_rate=None,
            # )
        # Load Pretrain
        # if backbone == "vit_base_patch8_384":
        #     path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        #     state_dict = torch.load(path, map_location="cpu")
        #     filtered_dict = checkpoint_filter_fn(state_dict, model)
        #     model.load_state_dict(filtered_dict, strict=True)
        # import ipdb; ipdb.set_trace()

        if "deit" in model_cfg['backbone']:
            # print_dict(default_cfg)
            load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
        else:
            load_custom_pretrained(model, default_cfg)
        return model

    def create_decoder(self, encoder, decoder_cfg):
        decoder_cfg = decoder_cfg.copy()
        if self.regress_module:
            decoder_cfg["n_cls"] = self.n_classes * 3
        else:
            decoder_cfg["n_cls"] = self.n_classes
        decoder_cfg["d_encoder"] = encoder.d_model
        decoder_cfg["patch_size"] = encoder.patch_size
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
        return decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


class Segmenter2(Segmenter):
    def __init__(self, config, reg=False, up_scale=False, *args, **kwargs):
        super(Segmenter2, self).__init__(config)
        self.config = config
        # self.n_cls = config['n_cls']
        self.patch_size = 16 # encoder.patch_size  #
        self.encoder = None # encoder
        self.decoder = None # decoder

        self.reg = reg
        self.model_cfg = config['model_cfg']
        self.decoder_cfg = config['decoder_cfg']
        # self.n_classes = config['dataset']['n_cls'] # config['special']['num_landmarks']

        self.up_scale = up_scale
        self.scale = 8
        self.n_classes = config['dataset']['n_cls'] if not up_scale else (self.scale ** 2) * config['dataset']['n_cls']
        self.regress_module = config['special']['regress']
        self.n_classes_output = config['dataset']['n_cls'] if not reg else config['dataset']['n_cls'] * 3
        self.original_cls = config['dataset']['n_cls']

        self.size = [384, 384]
        # if up_scale:
            # self.up_channel_conv = nn.Conv2d(self.n_classes, self.size[0]*self.size[1]*self.n_classes_output, kernel_size=1)
        # self.deconv = nn.ConvTranspose2d(16, 33, 3, stride=2, dilation=4)
        self.build()

    def forward(self, im, visualize=False, sigmoid=True, **kwargs):
        # H_ori, W_ori = im.size(2), im.size(3)
        H, W = im.size(2), im.size(3)
        masks = self.forward_to_masks(im, visualize=visualize)
        if self.up_scale:
            # masks2 = self.up_channel_conv(masks)
            # import ipdb; ipdb.set_trace()
            masks = rearrange(masks, "n (c ch cw ) h w -> n c (ch h) (cw w)", ch=self.scale, cw=self.scale)
        # masks = rearrange(masks2, "n (c ch cw ) h w -> n c (ch h) (cw w)", ch=self.size[0], cw=self.size[1])
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        if not self.reg:
            heatmap = masks[:, :self.n_classes, :, :]
            if sigmoid:
                heatmap = F.sigmoid(heatmap)
            return heatmap

        # import ipdb; ipdb.set_trace()
        #
        # masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        # masks = unpadding(masks, (H_ori, W_ori))
        #
        # if visualize:
        #     print("shape: ", masks.shape)
        #     save_image(masks[0].unsqueeze(1), "./tmp/masks_1.png")
        #
        # if not self.regress_module:
        #     # return masks
        #     return F.sigmoid(masks)
        #
        heatmap = F.sigmoid(masks[:, :self.n_classes, :, :])
        regression_x = masks[:, self.n_classes:2 * self.n_classes, :, :]
        regression_y = masks[:, 2 * self.n_classes:, :, :]
        return heatmap, regression_x, regression_y


def usage():
    from tutils import load_yaml
    config = load_yaml("configs/baseline/segmenter.yaml")
    model = Segmenter2(config)
    img = torch.ones((1,3,384,384))
    out = model(img)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    usage()