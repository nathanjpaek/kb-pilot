import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        out = -1.0 * out.sum(dim=1)
        return out.mean()


class IBLoss(nn.Module):

    def __init__(self, eta=1):
        super(IBLoss, self).__init__()
        self.eta = eta
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.entropy_loss = EntropyLoss()

    def forward(self, x, target):
        return self.cross_entropy_loss(x, target) + self.entropy_loss(x
            ) * self.eta


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
