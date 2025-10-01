import torch
import numpy as np
import torch.nn as nn


def orthog_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.num_outputs = num_outputs
        self.affine1 = orthog_layer_init(nn.Linear(num_inputs, 64))
        self.affine2 = orthog_layer_init(nn.Linear(64, 64))
        self.linear3 = orthog_layer_init(nn.Linear(64, num_outputs * 2),
            std=0.01)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        mu = self.linear3(x)[:, :self.num_outputs]
        log_std = self.linear3.bias[self.num_outputs:].unsqueeze(0).expand_as(
            mu)
        std = torch.exp(log_std)
        return mu, log_std, std


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_outputs': 4}]
