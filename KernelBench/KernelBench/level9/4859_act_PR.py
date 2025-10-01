import torch
import torch.nn as nn
import torch.utils.model_zoo


class act_PR(nn.Module):

    def __init__(self, affine=True):
        super(act_PR, self).__init__()
        self.prelu = nn.PReLU(num_parameters=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = (self.relu(x) + self.prelu(x)) / 2
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
