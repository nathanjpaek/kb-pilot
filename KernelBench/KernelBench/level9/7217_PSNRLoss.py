import torch
from torch import nn


class PSNRLoss(nn.Module):

    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def __repr__(self):
        return 'PSNR'

    def forward(self, output, target):
        mse = self.criterion(output, target)
        loss = 10 * torch.log10(1.0 / mse)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
