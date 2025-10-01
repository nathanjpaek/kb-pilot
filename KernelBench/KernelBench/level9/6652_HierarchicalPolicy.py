from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as f


class HierarchicalPolicy(nn.Module):

    def __init__(self, args):
        super(HierarchicalPolicy, self).__init__()
        self.fc_1 = nn.Linear(args.state_shape, 128)
        self.fc_2 = nn.Linear(128, args.noise_dim)

    def forward(self, state):
        x = f.relu(self.fc_1(state))
        q = self.fc_2(x)
        prob = f.softmax(q, dim=-1)
        return prob


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(state_shape=4, noise_dim=4)}]
