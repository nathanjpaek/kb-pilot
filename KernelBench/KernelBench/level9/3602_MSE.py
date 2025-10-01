import torch
import torch.nn as nn


class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        loss = torch.mean(torch.pow(pred - gt, 2))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
