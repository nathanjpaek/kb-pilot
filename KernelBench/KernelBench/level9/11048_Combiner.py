import torch
import numpy as np
import torch.nn as nn


def FC(shape=None, init=None):
    if init is None:
        K = shape[-2]
        init = [torch.rand(shape) * 2 - 1]
        shape_bias = shape.copy()
        shape_bias[-2] = 1
        init.append(torch.rand(shape_bias) * 2 - 1)
    else:
        K = init[0].shape[-2]
    fc = nn.Parameter(init[0] * np.sqrt(1 / K))
    fc_bias = nn.Parameter(init[1] * np.sqrt(1 / K))
    return fc, fc_bias


class Combiner(nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim, L):
        super(Combiner, self).__init__()
        self.fc1_z, self.fc1_z_bias = FC([L, z_dim, rnn_dim])
        self.fc2_z = nn.Linear(rnn_dim, z_dim)
        self.fc21_z = nn.Linear(z_dim, z_dim)
        self.fc3_z = nn.Linear(rnn_dim, z_dim)
        self.fc31_z = nn.Linear(z_dim, z_dim)
        self.tanh = nn.PReLU()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(x_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, y_{t:T})
        """
        h = torch.matmul(z_t_1, self.fc1_z) + self.fc1_z_bias
        h_combined = 0.5 * (self.tanh(h).mean(dim=0) + self.tanh(h_rnn))
        loc = self.tanh(self.fc2_z(h_combined))
        loc = self.fc21_z(loc)
        scale = self.tanh(self.fc3_z(h_combined))
        scale = self.fc31_z(scale)
        return loc, scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4, 'rnn_dim': 4, 'L': 4}]
