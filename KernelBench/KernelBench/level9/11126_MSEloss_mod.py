import torch
import torch.nn as nn


class MSEloss_mod(nn.Module):

    def __init__(self):
        super(MSEloss_mod, self).__init__()

    def forward(self, y_pred, y_gt):
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0].permute(1, 0)
        y = y_gt[:, :, 1].permute(1, 0)
        out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
        lossVal = torch.sum(out) / (out.shape[0] * out.shape[1])
        return lossVal


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
