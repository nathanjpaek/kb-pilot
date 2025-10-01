import torch
import torch.nn as nn
import torch.fft


class AdaptiveInstanceNormalization(nn.Module):

    def and__init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def forward(self, x, mean, std):
        whitened_x = torch.nn.functional.instance_norm(x)
        return whitened_x * std + mean


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
