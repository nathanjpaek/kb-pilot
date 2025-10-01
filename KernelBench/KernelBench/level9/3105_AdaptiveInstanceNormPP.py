import torch
import torch.nn as nn
import torch.jit
import torch.nn


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, in_size, ada_size):
        super(AdaptiveInstanceNorm, self).__init__()
        self.scale = nn.Linear(ada_size, in_size)
        self.bias = nn.Linear(ada_size, in_size)

    def forward(self, inputs, style):
        in_view = inputs.view(inputs.size(0), inputs.size(1), 1, 1, -1)
        mean = in_view.mean(dim=-1)
        std = in_view.std(dim=-1)
        scale = self.scale(style).view(style.size(0), -1, 1, 1)
        bias = self.bias(style).view(style.size(0), -1, 1, 1)
        return scale * (inputs - mean) / (std + 1e-06) + bias


class AdaptiveInstanceNormPP(AdaptiveInstanceNorm):

    def __init__(self, in_size, ada_size):
        super(AdaptiveInstanceNormPP, self).__init__(in_size, ada_size)
        self.mean_scale = nn.Linear(ada_size, in_size)

    def forward(self, inputs, style):
        in_view = inputs.view(inputs.size(0), inputs.size(1), 1, 1, -1)
        mean = in_view.mean(dim=-1)
        mean_mean = mean.mean(dim=1, keepdim=True)
        std = in_view.std(dim=-1)
        mean_std = mean.std(dim=1, keepdim=True)
        scale = self.scale(style).view(style.size(0), -1, 1, 1)
        mean_scale = self.mean_scale(style).view(style.size(0), -1, 1, 1)
        bias = self.bias(style).view(style.size(0), -1, 1, 1)
        result = scale * (inputs - mean) / (std + 1e-06) + bias
        correction = mean_scale * (mean - mean_mean) / (mean_std + 1e-06)
        return result + correction


def get_inputs():
    return [torch.rand([4, 64, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'ada_size': 4}]
