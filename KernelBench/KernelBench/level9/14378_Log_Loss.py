import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel


class Log_Loss(nn.Module):

    def __init__(self):
        super(Log_Loss, self).__init__()

    def forward(self, ytrue, ypred):
        delta = ypred - ytrue
        return torch.mean(torch.log(torch.cosh(delta)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
