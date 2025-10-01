import torch
import torch.nn.functional as F
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, n_actions, input_len):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_len, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out_policy = nn.Linear(100, n_actions)
        self.out_value = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        policy = F.softmax(self.out_policy(x))
        value = self.out_value(x)
        return policy, value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_actions': 4, 'input_len': 4}]
