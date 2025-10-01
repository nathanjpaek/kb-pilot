import torch
import torch.nn as nn


class ScalePredictor(nn.Module):

    def __init__(self, nz, scale_lr_decay=0.2, scale_bias=1.0):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)
        self.scale_bias = scale_bias
        self.scale_lr_decay = scale_lr_decay

    def forward(self, feat):
        scale = self.scale_lr_decay * self.pred_layer.forward(feat
            ) + self.scale_bias
        return scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nz': 4}]
