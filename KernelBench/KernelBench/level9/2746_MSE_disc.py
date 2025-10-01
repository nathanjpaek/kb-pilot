import torch
import torch.nn as nn


class MSE_disc(nn.Module):

    def __init__(self, weight_list=None):
        super(MSE_disc, self).__init__()
        self.weight_list = weight_list

    def forward(self, x, labels):
        loss = (x - labels) ** 2
        if self.weight_list is not None:
            loss = loss * self.weight_list
        return loss.mean(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
