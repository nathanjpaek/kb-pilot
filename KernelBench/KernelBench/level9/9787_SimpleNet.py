import torch
import torch.nn as nn


class SimpleNet(nn.Module):

    def __init__(self, width, input_size, output_size, pool='max'):
        super(SimpleNet, self).__init__()
        self.pool = nn.MaxPool2d(width, stride=width
            ) if pool == 'max' else nn.AvgPool2d(width, stride=width)
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = x.permute(0, 3, 1, 2)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'width': 4, 'input_size': 4, 'output_size': 4}]
