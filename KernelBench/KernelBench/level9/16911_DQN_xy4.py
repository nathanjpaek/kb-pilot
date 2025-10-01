import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class DQN_xy4(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    """

    def __init__(self):
        super(DQN_xy4, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
