import torch
import torch.nn as nn
import torch.utils.model_zoo


class act_PRT(nn.Module):

    def __init__(self, affine=True):
        super(act_PRT, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.prelu = nn.PReLU(num_parameters=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = (self.relu(x) + self.prelu(x) + self.tanh(x)) / 3
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
