import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        """ Simple two-layer neural network.
        """
        super(NeuralNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim * 2
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.ac1 = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.ac2 = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.ac1(self.l1(x.float()))
        return self.ac2(self.l2(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
