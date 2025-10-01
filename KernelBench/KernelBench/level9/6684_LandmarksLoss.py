import torch
import numpy as np
import torch.nn as nn
import torch.utils.data


class WingLoss(nn.Module):

    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t == -1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (
            abs_diff - self.C)
        return y.sum()


class LandmarksLoss(nn.Module):

    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = WingLoss()
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred * mask, truel * mask)
        return loss / (torch.sum(mask) + 1e-13)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
