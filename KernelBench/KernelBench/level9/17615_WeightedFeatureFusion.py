import torch
import torch.nn as nn
from torchvision.models.resnet import *
import torch.utils.data


class WeightedFeatureFusion(nn.Module):

    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers
        self.weight = weight
        self.n = len(layers) + 1
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)
            x = x * w[0]
        nx = x.shape[1]
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[
                self.layers[i]]
            na = a.shape[1]
            if nx == na:
                x = x + a
            elif nx > na:
                x[:, :na] = x[:, :na] + a
            else:
                x = x + a[:, :nx]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])]


def get_init_inputs():
    return [[], {'layers': [4, 4]}]
