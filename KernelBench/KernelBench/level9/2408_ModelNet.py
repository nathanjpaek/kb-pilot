import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
import torch.utils.data


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)


class ModelNet(nn.Module):

    def __init__(self, observation_space, action_space, h1=500, h2=500):
        super(ModelNet, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0] + action_space.
            shape[0], h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, observation_space.shape[0])
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(weight_init)

    def forward(self, ob, ac):
        h = torch.cat([ob, ac], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'observation_space': torch.rand([4, 4]), 'action_space':
        torch.rand([4, 4])}]
