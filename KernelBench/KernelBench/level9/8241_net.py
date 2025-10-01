import torch
import torch.nn as nn
import torch.nn.functional as F


class net(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 30)
        self.fc1.weight.data.normal_(0, 1)
        self.fc2 = nn.Linear(30, 20)
        self.fc2.weight.data.normal_(0, 1)
        self.fc3 = nn.Linear(20, output_dim)
        self.fc3.weight.data.normal_(0, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        out = self.fc3(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
