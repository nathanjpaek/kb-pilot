import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class SimpleNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         """
        super(SimpleNet, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         """
        x = F.relu(self.fc_in(x))
        x = self.drop(x)
        for i in range(9):
            x = F.relu(self.fc_hidden(x))
            x = self.drop(x)
        x = self.fc_out(x)
        x = self.sig(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
