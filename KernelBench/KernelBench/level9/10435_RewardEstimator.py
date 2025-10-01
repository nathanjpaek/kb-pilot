import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_parameters_util_x(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data, 1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.GRUCell):
            for mm in module.parameters():
                if mm.data.ndimension() == 2:
                    nn.init.xavier_normal_(mm.data, 1)
                elif mm.data.ndimension() == 1:
                    mm.data.zero_()


class RewardEstimator(nn.Module):
    """Estimates the reward the agent will receive. Value used as a baseline in REINFORCE loss"""

    def __init__(self, hid_dim):
        super(RewardEstimator, self).__init__()
        self.hid_dim = hid_dim
        self.v1 = nn.Linear(hid_dim, math.ceil(hid_dim / 2))
        self.v2 = nn.Linear(math.ceil(hid_dim / 2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util_x(self)

    def forward(self, x):
        x = x.detach()
        x = F.relu(self.v1(x))
        x = self.v2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hid_dim': 4}]
