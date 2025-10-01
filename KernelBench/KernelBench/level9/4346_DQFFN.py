import torch
import torch.nn as nn
import torch.nn.functional as F


class DQFFN(nn.Module):

    def __init__(self, n):
        """
        Create Feed-forward Network with n dim input and n dim output
        """
        super(DQFFN, self).__init__()
        self.n = n
        self.l1 = nn.Linear(n * (n + 1) // 2, 2048)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, n)

    def forward(self, x):
        """
        input is of shape (batch_size, n, n)
        """
        upper_indices = torch.triu_indices(self.n, self.n)
        x = x[:, upper_indices[0], upper_indices[1]]
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4}]
