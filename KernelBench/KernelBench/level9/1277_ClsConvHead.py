import torch
import torch.nn as nn
import torch.nn.parallel


class ClsConvHead(nn.Module):
    """Global average pooling + conv head for classification
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
