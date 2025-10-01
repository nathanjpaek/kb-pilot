import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class DQN_hot1(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """

    def __init__(self, m, n, num_actions):
        super(DQN_hot1, self).__init__()
        self.fc1 = nn.Linear(m * n, 15)
        self.fc2 = nn.Linear(15, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'m': 4, 'n': 4, 'num_actions': 4}]
