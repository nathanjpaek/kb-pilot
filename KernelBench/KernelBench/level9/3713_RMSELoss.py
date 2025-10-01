import torch
import torch.nn as nn


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, inputs, targets):
        tmp = (inputs - targets) ** 2
        loss = torch.mean(tmp)
        return torch.sqrt(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
