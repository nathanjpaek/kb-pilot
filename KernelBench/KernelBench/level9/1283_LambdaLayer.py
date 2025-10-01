import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):

    def __init__(self, planes):
        super(LambdaLayer, self).__init__()
        self.planes = planes

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self
            .planes // 4), 'constant', 0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'planes': 4}]
