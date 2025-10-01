import torch
import torch.nn as nn
import torch.utils.data


class KLLoss(nn.Module):

    def __init__(self, size_average=False):
        super().__init__()
        self.size_average = size_average

    def forward(self, mu, logvar):
        loss = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
