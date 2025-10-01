import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableBias(nn.Module):

    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class BinaryLinear(nn.Module):

    def __init__(self, in_chn, out_chn, bias=False):
        super(BinaryLinear, self).__init__()
        self.shape = out_chn, in_chn
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001,
            requires_grad=True)
        self.bias = None
        if bias:
            self.bias = LearnableBias(out_chn)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach(
            ) - cliped_weights.detach() + cliped_weights
        y = F.linear(x, binary_weights)
        if self.bias:
            y = self.bias(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_chn': 4, 'out_chn': 4}]
