import torch
from torch import nn
from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder



class SSLModel(torch.nn.Module):
    def initialize(self):
        # init.initialize_decoder(self.decoder)
        # init.initialize_head(self.segmentation_head)
        # if self.classification_head is not None:
        #     init.initialize_head(self.classification_head)
        pass

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)
        features = self.encoder(x)
        for afeatures in features:
            print("Encoder: ", afeatures.shape)
        decoder_output = self.decoder.forward_list(*features)
        for fea in decoder_output:
            print("Decoder: ", fea.shape)
        return decoder_output
        # masks = self.segmentation_head(decoder_output)
        # if self.classification_head is not None:
        #     labels = self.classification_head(features[-1])
        #     return masks, labels
        # return masks

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()

        x = self.forward(x)
        return x


class Unet(SSLModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 32),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder_extend(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        length_embedding = 16
        self.trans_5 = nn.Conv2d(512, length_embedding, kernel_size=1, padding=0)
        self.trans_4 = nn.Conv2d(256, length_embedding, kernel_size=1, padding=0)
        self.trans_3 = nn.Conv2d(128, length_embedding, kernel_size=1, padding=0)
        self.trans_2 = nn.Conv2d(64, length_embedding, kernel_size=1, padding=0)
        self.trans_1 = nn.Conv2d(32, length_embedding, kernel_size=1, padding=0)
        # self.trans_0 = nn.Conv2d(32, length_embedding, kernel_size=1, padding=0)

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        b, c, h, w = x.shape
        if c > 1:
            x = x[:, 0, :, :].unsqueeze(1)

        self.check_input_shape(x)
        features = self.encoder(x)
        out = self.decoder.forward_list(*features)
        fea0, fea1, fea2, fea3, fea4 = out[0], out[1], out[2], out[3], out[4]
        fea0 = self.trans_5(fea0)
        fea1 = self.trans_4(fea1)
        fea2 = self.trans_3(fea2)
        fea3 = self.trans_2(fea3)
        fea4 = self.trans_1(fea4)
        return [fea0, fea1, fea2, fea3, fea4]


class UnetDecoder_extend(UnetDecoder):
    def __init__(self, *args, **kwargs):
        super(UnetDecoder_extend, self).__init__(*args, **kwargs)

    def forward_list(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]

        x = self.center(head)
        output_list = []
        for i, decoder_block in enumerate(self.blocks):
            output_list.append(x)
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return output_list


def test_model():
    model_cfg = {
        'ENCODER_NAME': 'resnet34',
        'ENCODER_WEIGHTS': 'imagenet',
        'DECODER_CHANNELS': [256,128,64,32,32],
        'IN_CHANNELS': 1,
    }

    model = Unet()
    testdata = torch.ones((4,1,384,384))
    output = model(testdata)
    for fea in output:
        print("test model: ", fea.shape)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    test_model()