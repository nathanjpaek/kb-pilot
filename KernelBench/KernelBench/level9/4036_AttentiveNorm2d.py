import torch
import torch.nn as nn
import torch.utils.data


class AttentiveNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, hidden_channels=32, eps=1e-05,
        momentum=0.1, track_running_stats=False):
        super(AttentiveNorm2d, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=False, track_running_stats=
            track_running_stats)
        self.gamma = nn.Parameter(torch.randn(hidden_channels, num_features))
        self.beta = nn.Parameter(torch.randn(hidden_channels, num_features))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, hidden_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = super(AttentiveNorm2d, self).forward(x)
        size = output.size()
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y)
        y = self.sigmoid(y)
        gamma = y @ self.gamma
        beta = y @ self.beta
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)
        return gamma * output + beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
