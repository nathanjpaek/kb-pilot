import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp_2layer(nn.Module):

    def __init__(self, in_ch, in_dim, width=1):
        super(mlp_2layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim * in_dim, 256 * width)
        self.fc2 = nn.Linear(256 * width, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'in_dim': 4}]
