import torch
import torch.nn as nn


class GroupNorm2D(nn.Module):

    def __init__(self, num_channels, num_groups=4, epsilon=1e-05):
        super(GroupNorm2D, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_channels // 4
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4
        [N, _C, H, W] = list(x.shape)
        out = torch.reshape(x, (N, self.num_groups, self.num_channels //
            self.num_groups, H, W))
        variance, mean = torch.var(out, dim=[2, 3, 4], unbiased=False,
            keepdim=True), torch.mean(out, dim=[2, 3, 4], keepdim=True)
        out = (out - mean) / torch.sqrt(variance + self.epsilon)
        out = out.view(N, self.num_channels, H, W)
        out = self.gamma.view([1, self.num_channels, 1, 1]
            ) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
