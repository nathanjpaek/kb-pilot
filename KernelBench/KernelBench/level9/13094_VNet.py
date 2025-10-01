import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
import torch.utils.data


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)


class VNet(nn.Module):

    def __init__(self, observation_space, h1=200, h2=100):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.apply(weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'observation_space': torch.rand([4, 4])}]
