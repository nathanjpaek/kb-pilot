import torch
import numpy as np
from torch import nn


class UpsamplerModel(nn.Module):

    def __init__(self, output_shape, factor):
        assert output_shape[0] % factor == 0
        assert output_shape[1] % factor == 0
        super(UpsamplerModel, self).__init__()
        self.output_shape = output_shape
        seed = np.ones((1, 1, output_shape[0] // factor, output_shape[1] //
            factor)) * 0.5
        self.sigmoid = nn.Sigmoid()
        self.seed = nn.Parameter(data=torch.FloatTensor(seed),
            requires_grad=True)

    def forward(self):
        return nn.functional.interpolate(self.sigmoid(self.seed), size=self
            .output_shape, mode='bilinear')


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'output_shape': [4, 4], 'factor': 4}]
