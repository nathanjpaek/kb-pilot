import torch
import torch.nn as nn
import torch.optim
import torch.autograd
import torch.nn
import torch.nn.init


class LayerNorm(nn.Module):

    def __init__(self, dim, mean=0.0, std=1.0, fixed=False, eps=1e-06, ball
        =False):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.ball = ball
        if fixed:
            self.target_mean = mean
            self.target_std = std
        else:
            self.target_mean = nn.Parameter(torch.empty(dim).fill_(mean))
            self.target_std = nn.Parameter(torch.empty(dim).fill_(std))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(torch.mean((x - mean).pow(2), dim=-1, keepdim=True
            ) + self.eps)
        if self.ball:
            std = std.clamp(1.0)
        return self.target_std * (x - mean) / std + self.target_mean


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
