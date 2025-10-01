import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.sparse


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.linear = nn.Linear(style_dim, in_channel * 2)
        self.linear.weight.data.normal_()
        self.linear.bias.data.zero_()
        self.linear.bias.data[:in_channel] = 1
        self.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.linear(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'style_dim': 4}]
