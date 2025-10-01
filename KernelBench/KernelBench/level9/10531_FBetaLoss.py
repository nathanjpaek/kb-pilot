import torch
import torch.nn as nn


class FBetaLoss(nn.Module):

    def __init__(self, beta=1):
        super(FBetaLoss, self).__init__()
        self.eps = 1e-08
        self.beta = beta
        self.beta2 = beta ** 2
        return

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        tp = (inputs * target).sum(dim=1)
        precision = tp.div(inputs.sum(dim=1).add(self.eps))
        recall = tp.div(target.sum(dim=1).add(self.eps))
        fbeta = torch.mean((precision * recall).div(precision.mul(self.
            beta2) + recall + self.eps).mul(1 + self.beta2))
        return 1 - fbeta


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
