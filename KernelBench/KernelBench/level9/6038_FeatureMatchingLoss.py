import torch
import torch.utils.data
import torch
from torch import nn


class FeatureMatchingLoss(nn.Module):

    def __init__(self, n_layers_D, num_D):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.n_layers_D = n_layers_D
        self.num_D = num_D

    def forward(self, fake, real):
        loss = 0
        feat_weights = 4.0 / (self.n_layers_D + 1)
        d_weights = 1.0 / self.num_D
        for i in range(self.num_D):
            for j in range(len(fake[i]) - 1):
                loss += feat_weights * d_weights * self.criterion(fake[i][j
                    ], real[i][j].detach())
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_layers_D': 1, 'num_D': 4}]
