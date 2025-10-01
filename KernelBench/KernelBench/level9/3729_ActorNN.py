import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def init_hidden(layer):
    """
    Initialize NN layers
    """
    input_size = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(input_size)
    return -lim, lim


class ActorNN(nn.Module):
    """
    Actor Class
    """

    def __init__(self, state_size, action_size, hidden_size1=512,
        hidden_size2=256):
        """
        Initialize parameters
        """
        super(ActorNN, self).__init__()
        self.state_size = state_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.action_size = action_size
        self.FC1 = nn.Linear(self.state_size, self.hidden_size1)
        self.FC2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.FC3 = nn.Linear(self.hidden_size2, self.action_size)
        self.reset_parameters()

    def forward(self, state):
        x = F.relu(self.FC1(state))
        x = F.relu(self.FC2(x))
        x = torch.tanh(self.FC3(x))
        return x

    def reset_parameters(self):
        self.FC1.weight.data.uniform_(*init_hidden(self.FC1))
        self.FC2.weight.data.uniform_(*init_hidden(self.FC2))
        self.FC3.weight.data.uniform_(-0.003, 0.003)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
