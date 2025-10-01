import torch
import torch.nn as nn
import torch.utils.data


class AvgLayer(nn.Module):

    def forward(self, input):
        return input.mean(3, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
