import torch
import torch.utils.data
import torch
import torch.nn as nn


class IRW_L1_Loss(nn.Module):

    def __init__(self, threshold):
        super(IRW_L1_Loss, self).__init__()
        self.threshold = threshold

    def forward(self, x, y, beta):
        beta = beta.view(len(x), 1, 1, 1)
        beta = torch.nn.functional.threshold(beta, self.threshold, 0.0)
        assert len(beta) == len(x)
        loss = torch.mean(torch.abs(beta * x - beta * y))
        return loss


def get_inputs():
    return [torch.rand([4, 1, 1, 1]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 1, 1, 1])]


def get_init_inputs():
    return [[], {'threshold': 4}]
