import torch
from torch import nn
import torch.utils.data
import torch.nn.init


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, real_out, fake_out):
        d_loss = 1 - real_out + fake_out
        return d_loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
