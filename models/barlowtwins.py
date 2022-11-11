from collections import OrderedDict
import os
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from .UNet3D import UNet3D, UNet3DEncoder_tmp


class BarlowProjector(nn.Module):
    def __init__(self):
        super(BarlowProjector, self).__init__()
        # self.projector = nn.Sequential(nn.Conv3d(64, 512, kernel_size=1, padding=0),
        #                                nn.BatchNorm3d(1024),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv3d(1024, 2048, kernel_size=1, padding=0),
        #                                nn.BatchNorm3d(2048),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv3d(2048, 4096, kernel_size=1, padding=0)
        #                                )
        self.projector = nn.Sequential(nn.Linear(64, 1024),
                                       nn.BatchNorm1d(1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 2048),
                                       nn.BatchNorm1d(2048),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(2048, 4096)
                                       )

    def forward(self, x):
        return self.projector(x)


class MyBarlowTwins(nn.Module):
    def __init__(self, config):
        super(MyBarlowTwins, self).__init__()
        self.config = config
        self.backbone = UNet3D(1,1, emb_len=config['special']['emb_len'], ret_fea=True, config=config)

    def forward(self, y1, y2):
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        return z1, z2


class Learner(nn.Module):
    def __init__(self, config, logger=None, is_train=True,**kwargs):
        super().__init__()
        self.net = UNet3D(1,1, emb_len=config['special']['emb_len'], ret_fea=True, config=config)
        self.projector = BarlowProjector()
        self.bn = nn.BatchNorm1d(4096, affine=False)
        self.batch_size = config['training']['batch_size']
        self.config = config
        self.logger = logger
        self.lambd = 0.0051

        self.cosfn = nn.CosineSimilarity(dim=1)
        self.is_train = is_train

    def forward(self, x1, x2=None, **kwargs):
        if self.is_train:
            z1 = self.net(x1)
            z2 = self.net(x2)
            return z1, z2
        else:
            assert x2 is None
            return self.net(x1)

    def training_step(self, data, batch_idx, **kwargs):
        img1 = data["crp_1"]
        img2 = data["crp_2"]
        p1 = data['point_crp_1']
        p2 = data['point_crp_2']

        y1_list, y2_list = self.forward(img1, img2)
        scale = 8
        f_1 = torch.stack([y1_list[0][[id], :, p1[id, 0]//scale, p1[id, 1]//scale, p1[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[0][[id], :, p2[id, 0]//scale, p2[id, 1]//scale, p2[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss_1 = self.loss_fn(z1, z2)
        sim1 = self.cosfn(f_1, f_2).mean()

        scale = 4
        f_1 = torch.stack([y1_list[1][[id], :, p1[id, 0]//scale, p1[id, 1]//scale, p1[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[1][[id], :, p2[id, 0]//scale, p2[id, 1]//scale, p2[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss_2 = self.loss_fn(z1, z2)
        sim2 = self.cosfn(f_1, f_2).mean()

        scale = 2
        f_1 = torch.stack([y1_list[2][[id], :, p1[id, 0]//scale, p1[id, 1]//scale, p1[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[2][[id], :, p2[id, 0]//scale, p2[id, 1]//scale, p2[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss_3 = self.loss_fn(z1, z2)
        sim3 = self.cosfn(f_1, f_2).mean()

        scale = 1
        f_1 = torch.stack([y1_list[3][[id], :, p1[id, 0]//scale, p1[id, 1]//scale, p1[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[3][[id], :, p2[id, 0]//scale, p2[id, 1]//scale, p2[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss_4 = self.loss_fn(z1, z2)
        sim4 = self.cosfn(f_1, f_2).mean()

        loss = loss_1 + loss_2 + loss_3 + loss_4

        return {'loss': loss, "loss1": loss_1, "loss2": loss_2, "loss3": loss_3, "loss4": loss_4,
                "sim1": sim1, "sim2": sim2, "sim3": sim3, "sim4": sim4}

    def loss_fn(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        # torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def on_before_zero_grad(self, **kwargs):
        pass

    def configure_optimizers(self, **kwargs):
        # param_weights = []
        # param_biases = []
        # for param in self.parameters():
        #     if param.ndim == 1:
        #         param_biases.append(param)
        #     else:
        #         param_weights.append(param)
        # parameters = [{'params': param_weights}, {'params': param_biases}]
        #
        # optimizer = LARS(parameters, lr=0, weight_decay=1e-6,
        #              weight_decay_filter=exclude_bias_and_norm,
        #              lars_adaptation_filter=exclude_bias_and_norm)
        optimizer = optim.Adam(params=self.parameters(), \
                           lr=self.config['optim']['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=self.config['optim']['weight_decay'])
        return {'optimizer': optimizer, "scheduler": None}

    def validation_step(self, data, batch_idx, **kwargs):
        img1 = data["crp_1"]
        img2 = data["crp_2"]
        p1 = data['point_crp_1']
        p2 = data['point_crp_2']

        print("debug: valid", img1.shape, img2.shape)
        y1_list, y2_list = self.forward(img1, img2)
        scale = 8
        f_1 = torch.stack([y1_list[0][[id], :, p1[id, 0] // scale, p1[id, 1] // scale, p1[id, 2] // scale] for id in
                           range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[0][[id], :, p2[id, 0] // scale, p2[id, 1] // scale, p2[id, 2] // scale] for id in
                           range(self.batch_size)]).squeeze()
        sim1 = self.cosfn(f_1, f_2)

        scale = 4
        f_1 = torch.stack([y1_list[1][[id], :, p1[id, 0] // scale, p1[id, 1] // scale, p1[id, 2] // scale] for id in
                           range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[1][[id], :, p2[id, 0] // scale, p2[id, 1] // scale, p2[id, 2] // scale] for id in
                           range(self.batch_size)]).squeeze()
        sim2 = self.cosfn(f_1, f_2)

        scale = 2
        f_1 = torch.stack([y1_list[2][[id], :, p1[id, 0] // scale, p1[id, 1] // scale, p1[id, 2] // scale] for id in
                           range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[2][[id], :, p2[id, 0] // scale, p2[id, 1] // scale, p2[id, 2] // scale] for id in
                           range(self.batch_size)]).squeeze()
        sim3 = self.cosfn(f_1, f_2)

        scale = 1
        f_1 = torch.stack([y1_list[3][[id], :, p1[id, 0] // scale, p1[id, 1] // scale, p1[id, 2] // scale] for id in
                           range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[3][[id], :, p2[id, 0] // scale, p2[id, 1] // scale, p2[id, 2] // scale] for id in
                           range(self.batch_size)]).squeeze()
        sim4 = self.cosfn(f_1, f_2)

        return {"sim1": sim1, "sim2": sim2, "sim3": sim3, "sim4": sim4}


class Learner2(nn.Module):
    """ Learner2 with UNet3DEncoder as backbone network"""
    def __init__(self, config, logger=None, is_train=True, **kwargs):
        super().__init__()
        self.net = UNet3DEncoder_tmp(1,1, emb_len=config['special']['emb_len'])
        self.projector = BarlowProjector()
        self.bn = nn.BatchNorm1d(4096, affine=False)
        self.batch_size = config['training']['batch_size']
        self.config = config
        self.logger = logger
        self.lambd = 0.0051

        self.cosfn = nn.CosineSimilarity(dim=1)
        self.is_train = is_train

    def forward(self, x1, x2, **kwargs):
        if self.is_train:
            z1 = self.net(x1)
            z2 = self.net(x2)
            return z1, z2
        else:
            assert x2 is None
            return self.net(x1)

    def training_step(self, data, batch_idx, **kwargs):
        img1 = data["crp_1"]
        img2 = data["crp_2"]
        p1 = data['point_crp_1']
        p2 = data['point_crp_2']

        y1_list, y2_list = self.forward(img1, img2)
        scale = 8
        f_1 = torch.stack([y1_list[0][[id], :, p1[id, 0]//scale, p1[id, 1]//scale, p1[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[0][[id], :, p2[id, 0]//scale, p2[id, 1]//scale, p2[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss_1 = self.loss_fn(z1, z2)
        sim1 = self.cosfn(f_1, f_2).mean()

        scale = 4
        f_1 = torch.stack([y1_list[1][[id], :, p1[id, 0]//scale, p1[id, 1]//scale, p1[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[1][[id], :, p2[id, 0]//scale, p2[id, 1]//scale, p2[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss_2 = self.loss_fn(z1, z2)
        sim2 = self.cosfn(f_1, f_2).mean()

        scale = 2
        f_1 = torch.stack([y1_list[2][[id], :, p1[id, 0]//scale, p1[id, 1]//scale, p1[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[2][[id], :, p2[id, 0]//scale, p2[id, 1]//scale, p2[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss_3 = self.loss_fn(z1, z2)
        sim3 = self.cosfn(f_1, f_2).mean()

        scale = 1
        f_1 = torch.stack([y1_list[3][[id], :, p1[id, 0]//scale, p1[id, 1]//scale, p1[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        f_2 = torch.stack([y2_list[3][[id], :, p2[id, 0]//scale, p2[id, 1]//scale, p2[id, 2]//scale] for id in range(self.batch_size)]).squeeze()
        z1, z2 = self.projector(f_1), self.projector(f_2)
        loss_4 = self.loss_fn(z1, z2)
        sim4 = self.cosfn(f_1, f_2).mean()

        loss = loss_1 + loss_2 + loss_3 + loss_4

        return {'loss': loss, "loss1": loss_1, "loss2": loss_2, "loss3": loss_3, "loss4": loss_4,
                "sim1": sim1, "sim2": sim2, "sim3": sim3, "sim4": sim4}

    def loss_fn(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        # torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def on_before_zero_grad(self, **kwargs):
        pass

    def configure_optimizers(self, **kwargs):
        # param_weights = []
        # param_biases = []
        # for param in self.parameters():
        #     if param.ndim == 1:
        #         param_biases.append(param)
        #     else:
        #         param_weights.append(param)
        # parameters = [{'params': param_weights}, {'params': param_biases}]
        #
        # optimizer = LARS(parameters, lr=0, weight_decay=1e-6,
        #              weight_decay_filter=exclude_bias_and_norm,
        #              lars_adaptation_filter=exclude_bias_and_norm)
        optimizer = optim.Adam(params=self.parameters(), \
                           lr=self.config['optim']['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=self.config['optim']['weight_decay'])
        return {'optimizer': optimizer, "scheduler": None}

    def validation_step(self, data, batch_idx, **kwargs):
        raise NotImplementedError

    def load_ckpt(self):
        return self._load_unet_encoder()

    def _load_unet_encoder(self):
        """  Load Unet3DEncoder only """
        ckpt = self.config['base_dir'] + self.config['tag'] + f"/model_epoch_{self.config['pepoch']}.pth"
        assert os.path.exists(ckpt), f"{ckpt}"
        self.logger.info(f'Load CKPT {ckpt}')
        state_dict = torch.load(ckpt)
        # -----
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith("module."):
                print("skip: ", k)
                continue
            name = k.replace("module.", "")
            new_state_dict[name] = v
            # print("dict name: ", name)
        self.net.load_state_dict(new_state_dict)

    def testing_step(self, data, batch_idx, **kwargs):
        img1 = data["crp_1"]
        img2 = data["crp_2"]
        p1 = data['point_crp_1']
        p2 = data['point_crp_2']

        # print("debug: valid", img1.shape, img2.shape)
        # y1_list, y2_list = self.forward(img1, img2)
        # scale = 8
        # f_1 = torch.stack([y1_list[0][[id], :, p1[id, 0] // scale, p1[id, 1] // scale, p1[id, 2] // scale] for id in
        #                    range(self.batch_size)]).squeeze()
        # f_2 = torch.stack([y2_list[0][[id], :, p2[id, 0] // scale, p2[id, 1] // scale, p2[id, 2] // scale] for id in
        #                    range(self.batch_size)]).squeeze()
        # sim1 = self.cosfn(f_1, f_2)
        #
        # scale = 4
        # f_1 = torch.stack([y1_list[1][[id], :, p1[id, 0] // scale, p1[id, 1] // scale, p1[id, 2] // scale] for id in
        #                    range(self.batch_size)]).squeeze()
        # f_2 = torch.stack([y2_list[1][[id], :, p2[id, 0] // scale, p2[id, 1] // scale, p2[id, 2] // scale] for id in
        #                    range(self.batch_size)]).squeeze()
        # sim2 = self.cosfn(f_1, f_2)
        #
        # scale = 2
        # f_1 = torch.stack([y1_list[2][[id], :, p1[id, 0] // scale, p1[id, 1] // scale, p1[id, 2] // scale] for id in
        #                    range(self.batch_size)]).squeeze()
        # f_2 = torch.stack([y2_list[2][[id], :, p2[id, 0] // scale, p2[id, 1] // scale, p2[id, 2] // scale] for id in
        #                    range(self.batch_size)]).squeeze()
        # sim3 = self.cosfn(f_1, f_2)
        #
        # scale = 1
        # f_1 = torch.stack([y1_list[3][[id], :, p1[id, 0] // scale, p1[id, 1] // scale, p1[id, 2] // scale] for id in
        #                    range(self.batch_size)]).squeeze()
        # f_2 = torch.stack([y2_list[3][[id], :, p2[id, 0] // scale, p2[id, 1] // scale, p2[id, 2] // scale] for id in
        #                    range(self.batch_size)]).squeeze()
        # sim4 = self.cosfn(f_1, f_2)

        # return {"sim1": sim1, "sim2": sim2, "sim3": sim3, "sim4": sim4}
        record_dict = {}
        return record_dict


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def exclude_bias_and_norm(p):
    return p.ndim == 1


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


# class BarlowTwins(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.backbone = torchvision.models.resnet50(zero_init_residual=True)
#         self.backbone.fc = nn.Identity()
#
#         # projector
#         sizes = [2048] + list(map(int, args.projector.split('-')))
#         layers = []
#         for i in range(len(sizes) - 2):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
#             layers.append(nn.BatchNorm1d(sizes[i + 1]))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
#         self.projector = nn.Sequential(*layers)
#
#         # normalization layer for the representations z1 and z2
#         self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
#
#     def forward(self, y1, y2):
#         z1 = self.projector(self.backbone(y1))
#         z2 = self.projector(self.backbone(y2))
#
#         # empirical cross-correlation matrix
#         c = self.bn(z1).T @ self.bn(z2)
#
#         # sum the cross-correlation matrix between all gpus
#         c.div_(self.args.batch_size)
#         torch.distributed.all_reduce(c)
#
#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#         off_diag = off_diagonal(c).pow_(2).sum()
#         loss = on_diag + self.args.lambd * off_diag
#         return loss