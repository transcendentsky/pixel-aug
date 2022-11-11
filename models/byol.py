import copy
import random
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

from .utils import RandomApply, default, EMA, get_module_device, singleton, set_requires_grad, update_moving_average, loss_fn
from .netwrapper2d import NetWrapper2d, MLP2d
from einops import rearrange, repeat

# DEFAULT_AUG = torch.nn.Sequential(
#     RandomApply(
#         T.ColorJitter(0.8, 0.8, 0.8, 0.2),
#         p = 0.3
#     ),
#     T.RandomGrayscale(p=0.2),
#     T.RandomHorizontalFlip(),
#     RandomApply(
#         T.GaussianBlur((3, 3), (1.0, 2.0)),
#         p = 0.2
#     ),
#     T.RandomResizedCrop((image_size, image_size)),
#     T.Normalize(
#         mean=torch.tensor([0.485, 0.456, 0.406]),
#         std=torch.tensor([0.229, 0.224, 0.225])),
# )

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        projection_size=64,
        projection_hidden_size=256,
        moving_average_decay=0.99,
        use_momentum=True,
        forward_func=4,
    ):
        super().__init__()
        self.net = net
        self.forward_func = forward_func
        if forward_func > 1:
            print("WRN Not the Standard BYOL forward function! ")

        # if forward_func == 1:
        #     self.online_encoder = NetWrapper2(net, projection_size, projection_hidden_size)
        # elif forward_func in [2, 3, 4]:
        self.online_encoder = NetWrapper2d(net, projection_size, projection_hidden_size)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP2d(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward([torch.randn(2, 3, 192, 192, device=device),
                      torch.randn(2, 3, 192, 192, device=device),
                      torch.randn(2, 2, device=device),
                      torch.randn(2, 2, device=device)])

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        return_embedding=False,
        return_projection=True
    ):
        assert type(x) is list, f"Got {type(x)}"
        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        return self.forward_2d_1(x)

    def forward_2d_1(self, x):
        image_one, image_two, p1, p2 = x[0], x[1], x[2], x[3]
        online_proj_one, _ = self.online_encoder.forward(image_one, p1) # list of 4 items
        online_proj_two, _ = self.online_encoder.forward(image_two, p2)

        online_pred_one = [self.online_predictor(proj) for proj in online_proj_one]
        online_pred_two = [self.online_predictor(proj) for proj in online_proj_two]

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder.forward(image_one, p1)
            target_proj_two, _ = target_encoder.forward(image_two, p2)
            target_proj_one = [proj.detach() for proj in target_proj_one]
            target_proj_two = [proj.detach() for proj in target_proj_two]

        return online_pred_one, online_pred_two, target_proj_one, target_proj_two, online_proj_one, online_proj_two




# class BYOL_bk(nn.Module):
#     def __init__(
#         self,
#         net,
#         image_size,
#         hidden_layer = -2,
#         projection_size = 256,
#         projection_hidden_size = 4096,
#         augment_fn = None,
#         augment_fn2 = None,
#         moving_average_decay = 0.99,
#         use_momentum = True
#     ):
#         super().__init__()
#         self.net = net
#
#         # default SimCLR augmentation
#
#         DEFAULT_AUG = torch.nn.Sequential(
#             RandomApply(
#                 T.ColorJitter(0.8, 0.8, 0.8, 0.2),
#                 p = 0.3
#             ),
#             T.RandomGrayscale(p=0.2),
#             T.RandomHorizontalFlip(),
#             RandomApply(
#                 T.GaussianBlur((3, 3), (1.0, 2.0)),
#                 p = 0.2
#             ),
#             T.RandomResizedCrop((image_size, image_size)),
#             T.Normalize(
#                 mean=torch.tensor([0.485, 0.456, 0.406]),
#                 std=torch.tensor([0.229, 0.224, 0.225])),
#         )
#
#         self.augment1 = default(augment_fn, DEFAULT_AUG)
#         self.augment2 = default(augment_fn2, self.augment1)
#
#         self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
#
#         self.use_momentum = use_momentum
#         self.target_encoder = None
#         self.target_ema_updater = EMA(moving_average_decay)
#
#         self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
#
#         # get device of network and make wrapper same device
#         device = get_module_device(net)
#         self.to(device)
#
#         # send a mock image tensor to instantiate singleton parameters
#         self.forward(torch.randn(2, 3, image_size, image_size, device=device))
#
#     @singleton('target_encoder')
#     def _get_target_encoder(self):
#         target_encoder = copy.deepcopy(self.online_encoder)
#         set_requires_grad(target_encoder, False)
#         return target_encoder
#
#     def reset_moving_average(self):
#         del self.target_encoder
#         self.target_encoder = None
#
#     def update_moving_average(self):
#         assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
#         assert self.target_encoder is not None, 'target encoder has not been created yet'
#         update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
#
#     def forward(
#         self,
#         x,
#         return_embedding = False,
#         return_projection = True
#     ):
#         if return_embedding:
#             return self.online_encoder(x, return_projection = return_projection)
#
#         image_one, image_two = self.augment1(x), self.augment2(x)
#
#         online_proj_one, _ = self.online_encoder(image_one)
#         online_proj_two, _ = self.online_encoder(image_two)
#
#         online_pred_one = self.online_predictor(online_proj_one)
#         online_pred_two = self.online_predictor(online_proj_two)
#
#         with torch.no_grad():
#             target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
#             target_proj_one, _ = target_encoder(image_one)
#             target_proj_two, _ = target_encoder(image_two)
#             target_proj_one.detach_()
#             target_proj_two.detach_()
#
#         loss_one = loss_fn(online_pred_one, target_proj_two.detach())
#         loss_two = loss_fn(online_pred_two, target_proj_one.detach())
#
#         loss = loss_one + loss_two
#         return loss.mean()
#
