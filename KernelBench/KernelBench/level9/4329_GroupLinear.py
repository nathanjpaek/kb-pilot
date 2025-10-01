import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as f


class GroupLinear(nn.Module):

    def __init__(self, groups, channels, map_size, dropout=None):
        super(GroupLinear, self).__init__()
        self.groups = groups
        self.channels = channels
        self.map_size = map_size
        self.linear_nodes = int(map_size[0] * map_size[1] * channels / groups)
        check = map_size[0] * map_size[1] * channels % groups
        if check != 0:
            raise Exception('Invalid parameters for GroupLinear')
        self.fc = nn.Linear(self.linear_nodes, self.linear_nodes)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        return

    def forward(self, x):
        x = x.view([x.size()[0], self.groups, self.linear_nodes])
        x = self.fc(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view([x.size()[0], self.channels, self.map_size[0], self.
            map_size[1]])
        x = f.leaky_relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'groups': 4, 'channels': 4, 'map_size': [4, 4]}]
