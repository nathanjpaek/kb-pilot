import torch
import torch.nn as nn
import torch.nn.functional as F


class resnet_block(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(resnet_block, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in == self.dim_out:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1,
                padding=1, bias=False)
            self.bn_1 = nn.InstanceNorm2d(self.dim_out)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1,
                padding=1, bias=False)
            self.bn_2 = nn.InstanceNorm2d(self.dim_out)
        else:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2,
                padding=1, bias=False)
            self.bn_1 = nn.InstanceNorm2d(self.dim_out)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1,
                padding=1, bias=False)
            self.bn_2 = nn.InstanceNorm2d(self.dim_out)
            self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2,
                padding=0, bias=False)
            self.bn_s = nn.InstanceNorm2d(self.dim_out)

    def forward(self, input):
        if self.dim_in == self.dim_out:
            output = self.bn_1(self.conv_1(input))
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
            output = self.bn_2(self.conv_2(output))
            output = output + input
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
        else:
            output = self.bn_1(self.conv_1(input))
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
            output = self.bn_2(self.conv_2(output))
            input_ = self.bn_s(self.conv_s(input))
            output = output + input_
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
