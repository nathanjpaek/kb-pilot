import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class DQN_xy5(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a one hot board representation
    
    Params
    ------
    m, n: int
        Board size 
    num_actions: int 
        Number of action-value to output, one-to-one 
        correspondence to action in game.    
    """

    def __init__(self):
        super(DQN_xy5, self).__init__()
        self.fc1 = nn.Linear(4, 1000)
        self.fc2 = nn.Linear(1000, 2000)
        self.fc3 = nn.Linear(2000, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
