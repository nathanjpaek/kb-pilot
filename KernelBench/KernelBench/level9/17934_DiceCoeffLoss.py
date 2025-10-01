import torch
import torch.nn as nn


class DiceCoeffLoss(nn.Module):

    def __init__(self, eps: 'float'=0.0001):
        super(DiceCoeffLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0
            ], 'Predict and target must be same shape'
        inter = torch.dot(predict.view(-1), target.view(-1))
        union = torch.sum(predict) + torch.sum(target) + self.eps
        t = (2 * inter.float() + self.eps) / union.float()
        return t


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
