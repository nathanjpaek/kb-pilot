import torch
import torch.nn.functional as F
import torch.nn as nn


class qy(nn.Module):

    def __init__(self, d_dim, x_dim, y_dim, z_dim):
        super(qy, self).__init__()
        self.fc1 = nn.Linear(z_dim, y_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)
        return loc_y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_dim': 4, 'x_dim': 4, 'y_dim': 4, 'z_dim': 4}]
