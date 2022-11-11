import torch
import torchvision
import numpy as np
from tutils import tfilename, trans_args, trans_init, print_dict
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn as nn
from typing import Tuple
# import torch.nn.Tensor as Tensor
from fasteraa_model import *

Tensor = torch.Tensor


class AdvTrainer:
    # acknowledge https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py

    def __init__(self, model,
                    optimizer,
                    lossfn,
                    callbacks=c,
                    cfg=cfg.model,
                    use_cuda_nonblocking=True) -> None:
        super().__init__()
        self.model = model
        self.optim = optimizer
        self.lossfn = lossfn


    def forward(self,  data):
        # input: [-1, 1]
        input, target = data
        b = input.size(0) // 2
        a_input, a_target = input[:b], target[:b]
        n_input, n_target = input[b:], target[b:]
        loss, d_loss, a_loss = self.wgan_loss(n_input, n_target, a_input, a_target)

        return loss, d_loss, a_loss

    def wgan_loss(self,
                  n_input: Tensor,
                  n_target: Tensor,
                  a_input: Tensor,
                  a_target: Tensor
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        ones = n_input.new_tensor(1.0)
        self.model['main'].requires_grad_(True)
        self.model['main'].zero_grad()
        # real images
        output, n_output = self.model['main'](n_input)
        loss = self.cfg.cls_factor * F.cross_entropy(output, n_target)
        loss.backward(retain_graph=True)
        d_n_loss = n_output.mean()
        d_n_loss.backward(-ones)

        # augmented images
        with torch.no_grad():
            # a_input [-1, 1] -> [0, 1]
            a_input = self.model['policy'].denormalize_(a_input)
            augmented = self.model['policy'](a_input)

        _, a_output = self.model['main'](augmented)
        d_a_loss = a_output.mean()
        d_a_loss.backward(ones)
        gp = self.cfg.gp_factor * self.gradient_penalty(n_input, augmented)
        gp.backward()
        self.optimizer['main'].step()

        # train policy
        self.model['main'].requires_grad_(False)
        self.model['policy'].zero_grad()
        _output, a_output = self.model['main'](self.model['policy'](a_input))
        _loss = self.cfg.cls_factor * F.cross_entropy(_output, a_target)
        _loss.backward(retain_graph=True)
        a_loss = a_output.mean()
        a_loss.backward(-ones)
        self.optimizer['policy'].step()

        return loss + _loss, -d_n_loss + d_a_loss + gp, -a_loss

    def gradient_penalty(self,
                         real: Tensor,
                         fake: Tensor
                         ) -> Tensor:
        alpha = real.new_empty(real.size(0), 1, 1, 1).uniform_(0, 1)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_()
        _, output = self.model['main'](interpolated)
        grad = torch.autograd.grad(outputs=output, inputs=interpolated, grad_outputs=torch.ones_like(output),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        return (grad.norm(2, dim=1) - 1).pow(2).mean()


def training_function(logger, config):

    AdvTrainer(model,
                    optimizer,
                    F.cross_entropy,
                    callbacks=c,
                    cfg=cfg.model,
                    use_cuda_nonblocking=True)

    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='/home1/quanquan/datasets/cifar/cifar10', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='/home1/quanquan/datasets/cifar/cifar10',
                                            train=False, 
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)

    device = "cuda"
    learning_rate = 0.01
    curr_lr = learning_rate
    num_epochs = 100
    total_step = len(train_loader)
    model = resnet50().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Decay learning rate
        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    pass




# def training_function(config):
#     # Hyperparameters
#     alpha, beta = config["alpha"], config["beta"]
#     for step in range(10):
#         # Iterative training function - can be any arbitrary training procedure.
#         intermediate_score = objective(step, alpha, beta)
#         # Feed the score back back to Tune.
#         tune.report(mean_loss=intermediate_score)


def main(logger, config):
    print()

    training_function(logger, config)
    # analysis = tune.run(
    #     training_function,
    #     config={
    #         "alpha": tune.grid_search([0.001, 0.01, 0.1]),
    #         "beta": tune.choice([1, 2, 3])
    #     })

    # print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))


if __name__ == '__main__':
    import argparse

    args = trans_args()
    logger, config = trans_init(args)
    print_dict(config)
    main(logger, config)
    # eval(args.func)(logger, config)