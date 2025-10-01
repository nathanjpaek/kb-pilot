import torch
import torch.nn as nn


class ABS_cont(nn.Module):

    def __init__(self, theta=1 / 10):
        super(ABS_cont, self).__init__()
        self.theta = theta

    def forward(self, x, labels):
        loss = torch.abs(x - labels)
        mask = loss.gt(self.theta).float()
        loss = loss * mask
        return loss.mean(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
