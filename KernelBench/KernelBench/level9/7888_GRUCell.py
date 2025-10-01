import torch
import torch.nn as nn
import torch.utils.checkpoint


class GRUCell(nn.Module):

    def __init__(self, x_dim, h_dim):
        super(GRUCell, self).__init__()
        self.r = nn.Linear(x_dim + h_dim, h_dim, True)
        self.z = nn.Linear(x_dim + h_dim, h_dim, True)
        self.c = nn.Linear(x_dim, h_dim, True)
        self.u = nn.Linear(h_dim, h_dim, True)

    def forward(self, x, h):
        rz_input = torch.cat((x, h), -1)
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))
        u = torch.tanh(self.c(x) + r * self.u(h))
        new_h = z * h + (1 - z) * u
        return new_h


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'x_dim': 4, 'h_dim': 4}]
