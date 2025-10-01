import torch
import torch.nn as nn


class LWS(nn.Module):

    def __init__(self, num_features, num_classes, bias=True):
        super(LWS, self).__init__()
        self.fc = nn.Linear(num_features, num_classes, bias=bias)
        self.scales = nn.Parameter(torch.ones(num_classes))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        x *= self.scales
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4, 'num_classes': 4}]
