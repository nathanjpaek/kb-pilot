import torch
import torch.nn as nn


class my_Hingeloss(nn.Module):

    def __init__(self):
        super(my_Hingeloss, self).__init__()

    def forward(self, output, target):
        pos = torch.sum(output * target, 2)
        neg = torch.max((1 - target) * output, 2)
        loss = neg[0] - pos + 1
        loss[loss < 0] = 0
        loss = torch.mean(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
