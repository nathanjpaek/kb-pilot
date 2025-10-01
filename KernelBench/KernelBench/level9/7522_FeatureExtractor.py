import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(FeatureExtractor, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 48)
        self.affine3 = nn.Linear(48, num_outputs)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = torch.tanh(self.affine3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_outputs': 4}]
