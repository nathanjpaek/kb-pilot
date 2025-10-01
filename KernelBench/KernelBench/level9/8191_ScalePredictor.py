import torch
from torch import nn


class ScalePredictor(nn.Module):

    def __init__(self, nz):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        return scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nz': 4}]
