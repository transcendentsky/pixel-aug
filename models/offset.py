import torch
import torch.nn as nn
from einops import rearrange, repeat


class OffsetModule(nn.Module):
    def __init__(self, *args, **kwargs):
        dim = 64 * 2
        super(OffsetModule, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(dim, out_channels=dim, kernel_size=1),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(dim),
                                        nn.Conv2d(dim, out_channels=dim, kernel_size=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(dim, out_channels=2, kernel_size=1))

    def forward(self, x, x2):
        xx = torch.cat([x, x2], axis=1)
        d = self.conv_block(xx)
        return d

    def _process_patch_and_template(self, patch, template):
        h, w = patch.size()[-2], patch.size()[-1]
        template = repeat(template, "b c -> b c h w", h=h, w=w)
        emb = torch.cat([patch, template], axis=1)
        emb = self.conv_block(emb)
        return emb


class OffsetModule2(nn.Module):
    def __init__(self, *args, **kwargs):
        dim = 64 * 2
        super(OffsetModule2, self).__init__()
        self.conv_block = nn.Sequential(nn.Linear(dim, out_features=dim),
                                        nn.ReLU(inplace=True),
                                        # nn.BatchNorm2d(dim),
                                        nn.Linear(dim, out_features=dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(dim, out_features=2))

    def _process_vectors(self, v1, v2):
        # v1 = rearrange(v1, "b c -> b c 1 1")
        # v2 = rearrange(v2, "b c -> b c 1 1")
        emb = torch.cat([v1, v2], axis=1)
        print("debug: ", emb.shape)
        offset = self.conv_block(emb)
        return offset


def usage():
    offset_module = OffsetModule(128)
    offset_module2 = OffsetModule2(128)
    x1 = torch.ones((1,64,2,2))
    x2 = torch.ones((1,64,2,2))
    template = torch.ones(1,64)
    template2 = torch.ones(1, 64)
    out = offset_module(x1, x2)

    # out2 = offset_module._process_patch_and_template(x1, template)
    out3 = offset_module2._process_vectors(template2, template)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    usage()
