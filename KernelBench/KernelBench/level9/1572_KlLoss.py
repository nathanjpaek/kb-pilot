import torch
import torch.nn as nn


def kl_div(p: 'torch.Tensor', q: 'torch.Tensor') ->torch.Tensor:
    x = p * torch.log(p / q)
    return x.abs().mean()


class KlLoss(nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def forward(self, inputs: 'torch.Tensor', targets: 'torch.Tensor'):
        loss_kl = kl_div(targets, inputs)
        return loss_kl


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
