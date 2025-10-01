import torch
from torch import nn


class FastGlobalAvgPool2d:

    def __init__(self, flatten=False):
        self.flatten = flatten

    def __call__(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0),
                x.size(1), 1, 1)


class ECA(nn.Module):

    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = FastGlobalAvgPool2d(flatten=False)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1
            ) // 2, bias=False)

    def forward(self, x):
        squized_channels = self.avg_pool(x)
        channel_features = self.conv(squized_channels.squeeze(-1).transpose
            (-1, -2)).transpose(-1, -2).unsqueeze(-1)
        attention = torch.sigmoid(channel_features)
        return attention.expand_as(x) * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
