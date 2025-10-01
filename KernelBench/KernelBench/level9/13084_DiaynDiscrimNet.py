import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_
import torch.utils.data


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)


class DiaynDiscrimNet(nn.Module):

    def __init__(self, f_space, skill_space, h_size=300, discrim_f=lambda x: x
        ):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(f_space.shape[0], h_size)
        self.output_layer = nn.Linear(h_size, skill_space.shape[0])
        self.apply(weight_init)
        self.discrim_f = discrim_f

    def forward(self, ob):
        feat = self.discrim_f(ob)
        h = torch.relu(self.fc1(feat))
        return self.output_layer(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'f_space': torch.rand([4, 4]), 'skill_space': torch.rand([
        4, 4])}]
