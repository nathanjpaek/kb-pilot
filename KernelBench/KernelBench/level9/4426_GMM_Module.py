import math
import torch
import torch.nn as nn
import torch.utils.data


class GMM_Module(nn.Module):
    """
    GMM Module
    """

    def __init__(self, out_channel_M, k):
        super(GMM_Module, self).__init__()
        self.conv1 = nn.Conv2d(int(out_channel_M), k * out_channel_M,
            kernel_size=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2 * 
            1 * (k + 1) / (1 + 1)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.lrelu_1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(k * out_channel_M, 2 * k * out_channel_M,
            kernel_size=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2 * 
            1 * (k + 2 * k) / (k + k)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.lrelu_2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(2 * k * out_channel_M, 3 * k * out_channel_M,
            kernel_size=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2 * 
            1 * (2 * k + 3 * k) / (2 * k + 2 * k)))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, input):
        x = self.lrelu_1(self.conv1(input))
        x = self.lrelu_2(self.conv2(x))
        return self.conv3(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_channel_M': 4, 'k': 4}]
