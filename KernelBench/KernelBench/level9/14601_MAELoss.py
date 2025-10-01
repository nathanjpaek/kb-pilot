import torch
import torch.nn as nn


class MAELoss(nn.Module):

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(torch.abs(loss)) / torch.sum(val_pixels)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
