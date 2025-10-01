from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_z, config.d_z // 2)
        self.fc2 = nn.Linear(config.d_z // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(d_z=4)}]
