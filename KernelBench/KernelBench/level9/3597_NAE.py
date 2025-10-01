import torch
import torch.nn as nn


class NAE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        diff = torch.abs(pred - gt)
        loss = torch.mean(torch.abs(diff / gt))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
