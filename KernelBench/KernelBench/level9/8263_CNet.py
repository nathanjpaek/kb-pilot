import torch
import torch.nn as nn
import torch.nn.functional as F


class CNet(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x + y)
        actions_value = self.out(net)
        return actions_value


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
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'s_dim': 4, 'a_dim': 4}]
