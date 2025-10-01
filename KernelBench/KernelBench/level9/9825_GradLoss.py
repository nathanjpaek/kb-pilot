import torch
import torch.nn as nn
import torch.utils.data
import torch.optim


class GradLoss(nn.Module):

    def __init__(model):
        super(GradLoss, model).__init__()

    def forward(model, grad_fake, grad_real):
        return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
