import torch
import torch.nn as nn


class BhattacharyyaDistance(nn.Module):

    def __init__(self):
        super(BhattacharyyaDistance, self).__init__()

    def forward(self, hist1, hist2):
        bh_dist = torch.sqrt(hist1 * hist2).sum()
        return bh_dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
