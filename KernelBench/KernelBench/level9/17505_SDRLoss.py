import torch
import torch.nn as nn


class SDRLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, deg, clean):
        loss_sdr = -1.0 * torch.mean(deg * clean) ** 2 / (torch.mean(deg **
            2) + 2e-07)
        return loss_sdr


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
