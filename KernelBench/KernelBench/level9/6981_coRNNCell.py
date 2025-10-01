import torch
from torch import nn
import torch.nn.utils


class coRNNCell(nn.Module):

    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self, x, hy, hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy), 1))
            ) - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz
        return hy, hz


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_inp': 4, 'n_hid': 4, 'dt': 4, 'gamma': 4, 'epsilon': 4}]
