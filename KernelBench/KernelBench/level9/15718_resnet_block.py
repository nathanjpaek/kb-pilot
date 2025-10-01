import torch
import torch.nn as nn
import torch.nn.functional as F


class resnet_block(nn.Module):

    def __init__(self, ef_dim):
        super(resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1,
            padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1,
            padding=0, bias=True)

    def forward(self, input):
        output = self.conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.conv_2(output)
        output = output + input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ef_dim': 4}]
