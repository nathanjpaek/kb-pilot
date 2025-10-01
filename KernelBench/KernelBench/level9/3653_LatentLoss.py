import torch
from torch import Tensor
import torch.nn as nn


class LatentLoss(nn.Module):

    def forward(self, mu: 'Tensor', logvar: 'Tensor') ->Tensor:
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
