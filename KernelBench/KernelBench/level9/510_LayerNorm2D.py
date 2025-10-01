import torch
import torch.nn as nn


class LayerNorm2D(nn.Module):

    def __init__(self, num_channels, epsilon=1e-05):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert list(x.shape)[1] == self.num_channels
        assert len(x.shape) == 4
        variance, mean = torch.var(x, dim=[1, 2, 3], unbiased=False
            ), torch.mean(x, dim=[1, 2, 3])
        out = (x - mean.view([-1, 1, 1, 1])) / torch.sqrt(variance.view([-1,
            1, 1, 1]) + self.epsilon)
        out = self.gamma.view([1, self.num_channels, 1, 1]
            ) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
