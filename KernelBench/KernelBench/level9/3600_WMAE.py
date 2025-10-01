import torch
import torch.nn as nn


class WMAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = [300, 1, 200]

    def forward(self, pred, gt):
        diff = torch.abs(pred - gt)
        loss = 0
        for i in range(3):
            loss += torch.sum(diff[:, i] * self.weight[i])
        loss /= gt.size(0) * sum(self.weight)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
