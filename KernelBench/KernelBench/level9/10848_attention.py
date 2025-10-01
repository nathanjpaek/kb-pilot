import torch
import torch.utils
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


class attention(nn.Module):

    def __init__(self, input_channels, map_size):
        super(attention, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=map_size)
        self.fc1 = nn.Linear(in_features=input_channels, out_features=
            input_channels // 2)
        self.fc2 = nn.Linear(in_features=input_channels // 2, out_features=
            input_channels)

    def forward(self, x):
        output = self.pool(x)
        output = output.view(output.size()[0], output.size()[1])
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.sigmoid(output)
        output = output.view(output.size()[0], output.size()[1], 1, 1)
        output = torch.mul(x, output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'map_size': 4}]
