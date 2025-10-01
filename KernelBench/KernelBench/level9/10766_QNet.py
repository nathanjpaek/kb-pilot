import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_
from torch.nn.init import uniform_
import torch.nn.functional as F


def mini_weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(uniform_(m.weight.data, -0.003, 0.003))
        m.bias.data.fill_(0)


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)


class QNet(nn.Module):

    def __init__(self, ob_space, ac_space, h1=300, h2=400):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(ob_space.shape[0], h1)
        self.fc2 = nn.Linear(ac_space.shape[0] + h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(mini_weight_init)

    def forward(self, ob, ac):
        h = F.relu(self.fc1(ob))
        h = torch.cat([h, ac], dim=-1)
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ob_space': torch.rand([4, 4]), 'ac_space': torch.rand([4,
        4])}]
