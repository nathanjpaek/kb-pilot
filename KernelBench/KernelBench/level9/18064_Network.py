from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.l1 = nn.Linear(self.config['in_feature'], 500)
        self.l2 = nn.Linear(500, 350)
        self.l3 = nn.Linear(350, 200)
        self.l4 = nn.Linear(200, 130)
        self.l5 = nn.Linear(130, self.config['out_feature'])

    def forward(self, x):
        data = x.view(-1, self.config['in_feature'])
        y = F.relu(self.l1(data))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = F.relu(self.l4(y))
        return self.l5(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(in_feature=4, out_feature=4)}]
