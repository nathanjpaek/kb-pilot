import torch
import torch.nn as nn


class weightedFeatureFusion(nn.Module):

    def __init__(self, layers, weight=False):
        super(weightedFeatureFusion, self).__init__()
        self.layers = layers
        self.weight = weight
        self.n = len(layers) + 1
        if weight:
            self.w = torch.nn.Parameter(torch.zeros(self.n))

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)
            x = x * w[0]
        nc = x.shape[1]
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[
                self.layers[i]]
            ac = a.shape[1]
            dc = nc - ac
            if dc > 0:
                x[:, :ac] = x[:, :ac] + a
            elif dc < 0:
                x = x + a[:, :nc]
            else:
                x = x + a
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([5, 4, 4, 4])]


def get_init_inputs():
    return [[], {'layers': [4, 4]}]
