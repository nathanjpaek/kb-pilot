import torch
import torch.nn.functional as F
import torch.nn as nn


class REINFORCE(nn.Module):

    def __init__(self, input_size, num_actions):
        super(REINFORCE, self).__init__()
        self.fc = nn.Linear(input_size, 256)
        self.head = nn.Linear(256, num_actions)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)
        return F.softmax(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_actions': 4}]
