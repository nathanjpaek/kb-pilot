import torch
import torch.nn as nn
import torch.utils.model_zoo


class act_RT(nn.Module):

    def __init__(self, affine=True):
        super(act_RT, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = (self.relu(x) + self.tanh(x)) / 2
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
