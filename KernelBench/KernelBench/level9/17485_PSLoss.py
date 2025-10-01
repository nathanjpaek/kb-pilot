import torch
import torch.nn as nn
import torch.fft


class PSLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):
        x_power = torch.abs(torch.fft.fftn(x, dim=[2, 3]))
        y_power = torch.abs(torch.fft.fftn(y, dim=[2, 3]))
        loss = self.l1_loss(x_power, y_power).sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
