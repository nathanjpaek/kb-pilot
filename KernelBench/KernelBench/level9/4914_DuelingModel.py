import torch
import torch.nn as nn


class DuelingModel(nn.Module):

    def __init__(self, n_input, n_output, n_hidden):
        super(DuelingModel, self).__init__()
        self.adv1 = nn.Linear(n_input, n_hidden)
        self.adv2 = nn.Linear(n_hidden, n_output)
        self.val1 = nn.Linear(n_input, n_hidden)
        self.val2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        adv = nn.functional.relu(self.adv1(x))
        adv = self.adv2(adv)
        val = nn.functional.relu(self.val1(x))
        val = self.val2(val)
        return val + adv - adv.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input': 4, 'n_output': 4, 'n_hidden': 4}]
