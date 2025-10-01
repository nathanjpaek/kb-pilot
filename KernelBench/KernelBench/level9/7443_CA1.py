import torch
import torch.nn as nn
import torch.nn.functional as F


class CA1(nn.Module):
    """Reconstructs the inputs that originated from EC network.

    Consists of 2 fully connected layers, recieving inputs from CA3
    and outputs to EC. 
    """

    def __init__(self, N, D_in, D_out, resize_dim):
        super(CA1, self).__init__()
        self.N, self.resize_dim = N, resize_dim
        self.fc1 = nn.Linear(D_in, 100)
        self.fc2 = nn.Linear(100, D_out)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = x.view(self.N, 1, self.resize_dim, self.resize_dim)
        return x


def get_inputs():
    return [torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {'N': 4, 'D_in': 4, 'D_out': 4, 'resize_dim': 4}]
