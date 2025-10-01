import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class DQN_xy1(nn.Module):
    """
    A MLP for DQN learning. 
    
    Note: Uses a (x,y) coordinate board/action representation. 
    """

    def __init__(self):
        super(DQN_xy1, self).__init__()
        self.fc1 = nn.Linear(4, 15)
        self.fc2 = nn.Linear(15, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
