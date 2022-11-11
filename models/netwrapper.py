import torch.nn as nn
from .utils import singleton
import torch
from einops import rearrange , repeat

def flatten(t):
    return t.reshape(t.shape[0], -1)


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class MLP3d(nn.Module):
    def __init__(self, in_channels, out_channels=64, hidden_size=256):
        super(MLP3d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_size, kernel_size=1, padding=0),
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


class NetWrapper(nn.Module):
    """ NetWrapper for Unet ( with outputs of multi layers ) """
    def __init__(self, net, projection_size=64, projection_hidden_size=256, mode="train"):
        super().__init__()
        self.net = net
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.mode = mode
        self.projector_0 = None
        self.projector_1 = None
        self.projector_2 = None
        self.projector_3 = None
        self.projector_4 = None

    @singleton('projector_0')
    def _get_projector_0(self, hidden):
        dim = hidden.shape[1]
        projector_1 = MLP3d(dim, self.projection_size, self.projection_hidden_size)
        return projector_1.to(hidden)

    @singleton('projector_1')
    def _get_projector_1(self, hidden):
        dim = hidden.shape[1]
        projector_1 = MLP3d(dim, self.projection_size, self.projection_hidden_size)
        return projector_1.to(hidden)

    @singleton('projector_2')
    def _get_projector_2(self, hidden):
        dim = hidden.shape[1]
        projector_2 = MLP3d(dim, self.projection_size, self.projection_hidden_size)
        return projector_2.to(hidden)

    @singleton('projector_3')
    def _get_projector_3(self, hidden):
        dim = hidden.shape[1]
        projector_3 = MLP3d(dim, self.projection_size, self.projection_hidden_size)
        return projector_3.to(hidden)

    @singleton('projector_4')
    def _get_projector_4(self, hidden):
        dim = hidden.shape[1]
        projector_4 = MLP3d(dim, self.projection_size, self.projection_hidden_size)
        return projector_4.to(hidden)

    def get_representation(self, x):
        # print("debug represent", x.shape)
        return self.net(x)

    def forward(self, x, p):
        representation = self.get_representation(x)

        projector_0 = self._get_projector_0(representation[0])
        projector_1 = self._get_projector_1(representation[1])
        projector_2 = self._get_projector_2(representation[2])
        projector_3 = self._get_projector_3(representation[3])
        projector_4 = self._get_projector_4(representation[4])

        bs = representation[0].shape[0]
        p = p.cpu().detach().numpy().astype(int)
        channel = representation[0].shape[1]
        scale = 1*2
        f_proj = torch.stack(
            [representation[0][[i], :, p[i, 0]//scale, p[i, 1]//scale, p[i, 2]//scale] for i in range(bs)]).view((bs, channel))
        projection0 = projector_0(rearrange(f_proj, "b c -> b c 1 1 1"))
        scale = 2*2
        f_proj = torch.stack(
            [representation[1][[i], :, p[i, 0]//scale, p[i, 1]//scale, p[i, 2]//scale] for i in range(bs)]).view((bs, channel))
        projection1 = projector_1(rearrange(f_proj, "b c -> b c 1 1 1"))
        scale = 4*2
        f_proj = torch.stack(
            [representation[2][[i], :, p[i, 0]//scale, p[i, 1]//scale, p[i, 2]//scale] for i in range(bs)]).view((bs, channel))
        projection2 = projector_2(rearrange(f_proj, "b c -> b c 1 1 1"))
        scale = 8*2
        f_proj = torch.stack(
            [representation[3][[i], :, p[i, 0]//scale, p[i, 1]//scale, p[i, 2]//scale] for i in range(bs)]).view((bs, channel))
        projection3 = projector_3(rearrange(f_proj, "b c -> b c 1 1 1"))
        scale = 16*2
        f_proj = torch.stack(
            [representation[4][[i], :, p[i, 0]//scale, p[i, 1]//scale, p[i, 2]//scale] for i in range(bs)]).view((bs, channel))
        projection4 = projector_4(rearrange(f_proj, "b c -> b c 1 1 1"))

        projection = [projection0, projection1, projection2, projection3, projection4]
        return projection, representation

