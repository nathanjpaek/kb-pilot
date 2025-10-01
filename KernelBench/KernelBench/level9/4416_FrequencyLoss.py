import torch
import torch.nn as nn


class FrequencyLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=0.001):
        super(FrequencyLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, x, y):
        x_fft = torch.fft.rfft2(x, dim=(2, 3))
        y_fft = torch.fft.rfft2(y, dim=(2, 3))
        loss = self.criterion(x_fft, y_fft)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
