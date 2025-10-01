import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal


def ZeroInitializer(param):
    shape = param.size()
    init = np.zeros(shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def Linear(initializer=kaiming_normal, bias_initializer=ZeroInitializer):


    class CustomLinear(nn.Linear):

        def reset_parameters(self):
            initializer(self.weight)
            if self.bias is not None:
                bias_initializer(self.bias)
    return CustomLinear


class Lift(nn.Module):

    def __init__(self, in_features, out_features):
        super(Lift, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lift = Linear()(self.in_features, self.out_features * 2)

    def forward(self, input):
        return F.tanh(self.lift(input))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
