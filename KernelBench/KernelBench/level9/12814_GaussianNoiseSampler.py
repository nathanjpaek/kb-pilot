import torch
import torch as th
import torch.nn as nn


class GaussianNoiseSampler(nn.Module):

    def __init__(self, scale=0.01, inplace=False):
        super(GaussianNoiseSampler, self).__init__()
        if scale < 0:
            raise ValueError(
                'noise scale has to be greather than 0, but got {}'.format(
                scale))
        self.scale = scale
        self.inplace = inplace

    def forward(self, inputs):
        if self.scale:
            inputs.add_(th.randn_like(inputs) * self.scale)
        return inputs

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'scale={}{}'.format(self.scale, inplace_str)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
