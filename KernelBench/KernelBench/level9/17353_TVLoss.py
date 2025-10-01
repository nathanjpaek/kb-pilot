import torch
from torch import nn


class TVLoss(nn.Module):

    def __init__(self, tvloss_weight=1):
        super(TVLoss, self).__init__()
        self.tvloss_weight = tvloss_weight

    def forward(self, generated):
        b, c, h, w = generated.size()
        h_tv = torch.pow(generated[:, :, 1:, :] - generated[:, :, :h - 1, :], 2
            ).sum()
        w_tv = torch.pow(generated[:, :, :, 1:] - generated[:, :, :, :w - 1], 2
            ).sum()
        return self.tvloss_weight * (h_tv + w_tv) / (b * c * h * w)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
