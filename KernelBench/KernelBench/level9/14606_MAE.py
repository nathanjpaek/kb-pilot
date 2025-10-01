import torch
import torch.nn as nn


class MAE(nn.Module):

    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float() * (outputs > 0).float()
        err = torch.abs(target * val_pixels - outputs * val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1,
            keepdim=True)
        return torch.mean(loss / cnt) * 1000


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
